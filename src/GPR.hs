{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeApplications #-}

module GPR where

import           Data.List                          (isSuffixOf)
import           Data.Default.Class
import           Control.Monad                      (when, foldM)
import           Control.Monad.State
import           Torch                              ( Tensor, TensorOptions
                                                    , LearningRate, Parameter
                                                    , KeepDim (..), Dim (..)
                                                    , Diag (..), Tri (..)
                                                    , ScriptModule )
import qualified Torch                         as T
import qualified Torch.Functional.Internal     as T ( powScalar, powScalar'
                                                    , negative, hstack, vstack
                                                    , cartesian_prod )
import           Torch.Optim.CppOptim               (AdamOptions)
import qualified Torch.Optim.CppOptim          as T

import           System.IO.Unsafe                   (unsafePerformIO)

import           Lib
import           Plot

-- | Tensor Options
opts :: TensorOptions
opts = T.withDevice cpu . T.withDType T.Double $ T.defaultOpts
-- opts = T.withDevice gpu . T.withDType T.Double $ T.defaultOpts

-- | Gaussian Process State
data GPRState = GPRState { nr :: !Int          -- ^ Number of restarts of local optimizer
                         , op :: !AdamOptions  -- ^ Optimizer used during fitting
                         , α  :: !LearningRate -- ^ Optimizer Learning Rate
                         , x  :: !Tensor       -- ^ Training Data Inputs
                         , y  :: !Tensor       -- ^ Training Data Outputs
                         , k  :: !Tensor       -- ^ Correlation Matrix of training instances
                         , l  :: !Tensor       -- ^ Inverse of Correlation Matrix
                         , μ  :: !Tensor       -- ^ Estimation of Mean
                         , σ  :: !Tensor       -- ^ Estimation of Variance
                         , θ  :: !Parameter    -- ^ Logarithm of correlation length
                         , lb :: !Tensor       -- ^ Lower bound
                         , ub :: !Tensor       -- ^ Upper bound
                         } deriving (Show)

-- | Gaussian Kernel Function
gaussianKernel :: Tensor -> Tensor -> Tensor -> Tensor
gaussianKernel θ' x1 x2 = k'
  where
    [d,f] = T.shape x1
    d'    = head $ T.shape x2
    x1'   = T.reshape [d,d',f] $ T.repeatInterleaveScalar x1 d' 0
    k'    = T.exp . T.negative . T.sumDim (Dim 2) RemoveDim T.Double
          . T.mul θ' . flip T.powScalar 2.0 $ x1' - x2

-- | Negative Likelihood
-- nll :: T.Tensor -> State GPRState T.Tensor
nll :: Tensor -> StateT GPRState IO Tensor
nll θ' = do
    s@GPRState{..} <- get
    let θ''   = T.squeezeAll $ T.powScalar' 10 θ'
        n     = head $ T.shape x
        n'    = realToFrac n :: Float
        ones  = T.ones [ n , 1 ] opts
        ones' = T.transpose2D ones
        id'   = T.mulScalar (1.0e-12 :: Float) $ T.eyeSquare n opts
        k'    = gaussianKernel θ'' x x + id'
        l'    = cholesky' Lower k'
        μ'    = T.matmul ones' (T.choleskySolve Lower y l')
              / T.matmul ones' (T.choleskySolve Lower ones l')
        y''   = y - μ' * ones
        σ'    = T.divScalar n' . T.matmul (T.transpose2D y'')
              $ T.choleskySolve Lower y'' l'
        det'  = T.mulScalar @Double 2.0 . T.sumAll . T.log . T.abs
              . T.diag (Diag 0) $ l'
        -- det' = T.det l'
        nll'  = (T.negative . T.mulScalar (n' / 2.0) $ T.log σ')
              - T.mulScalar (0.5 :: Float) det'
        loss  = T.abs . T.negative . T.squeezeAll $ nll'
    put $ s { k = k', l = l', μ = μ', σ = σ' }
    pure loss

fit' :: [Parameter] -> StateT GPRState IO [(Tensor, Tensor)]
fit' [] = pure []
fit' (p:params) = do
    liftIO . putStrLn $ "Starting point " ++ show (length params) ++ ": " ++ show p
    GPRState{..} <- get
    o <- liftIO $ T.initOptimizer op p
    -- let o = T.mkAdam 0 0.9 0.999 [p]
    x' <- T.toDependent . fst <$> foldM (\(p',o') i -> do
        l' <- nll . clamp' lb ub $ T.toDependent p'
        when (mod i 100 == 0) . liftIO . putStrLn
            $ "\tNLL (" ++ show i ++ "): " ++ show (T.asValue l' :: Double)
        liftIO $ T.runStep p' o' l' α
        ) (p,o) [ 1 .. 2000 :: Int ]
    y' <- nll x'
    liftIO . putStrLn $ "Final point " ++ show (length params) ++ ": " ++ show x'
    ((x',y'):) <$> fit' params

fit :: StateT GPRState IO Tensor
fit = do
    GPRState{..} <- get
    let dims = last $ T.shape x
    lhd <- liftIO $ flip T.withTensorOptions opts <$> lhsMaxMin dims nr nr
    points' <- liftIO . mapM (`T.makeIndependentWithRequiresGrad` True)
                      . rows $ lhd -- scale' lb ub lhd
    opt <- fit' points'
    let θs  = T.stack (T.Dim 0) $ map fst opt
        ls  = T.cat (T.Dim 0) $ map (T.view [1] . snd) opt
        idx = T.argmin ls 0 False
        θ'  = T.squeezeAll $ T.indexSelect 0 idx θs
        θ'' = T.powScalar' 10 $ T.toDevice cpu θ'
    y' <- nll θ'
    liftIO . putStrLn $ "\nLowest Loss: " ++ show y'
    liftIO . putStrLn $ "Final θ: " ++ show θ'
    pure θ''

predict :: Tensor -> State GPRState (Tensor, Tensor)
predict x' = do
    GPRState{..} <- get
    let n    = head $ T.shape x
        ones = T.ones [n, 1] opts
        k'   = gaussianKernel (T.toDependent θ) x' x
        μ'   = T.add μ . T.matmul (T.transpose2D k)
             $ T.choleskySolve Lower (y - μ * ones) l
        σ'   = T.mul σ . (1.0 -) . T.diag (Diag 0) . T.matmul (T.transpose2D k')
             $ T.choleskySolve T.Lower k' l
    pure (μ', σ')

score :: Tensor -> Tensor -> State GPRState Tensor
score x' y' = T.sqrt . T.mean . T.powScalar' 2.0 . subtract y' . fst
            <$> predict x'

predict' :: Tensor -> Tensor -> Tensor -> Tensor -> Tensor -> Tensor -> Tensor
         -> (Tensor, Tensor)
predict' xt yt θ' μ' σ' l' x' = (T.squeezeAll μ'', T.squeezeAll σ'')
  where
    n     = head $ T.shape xt
    ones' = T.ones [n, 1] opts
    k'    = gaussianKernel θ' xt x'
    y'    = yt - μ' * ones'
    μ''   = T.add μ' . T.matmul (T.transpose2D k')
          $ T.choleskySolve Lower y' l'
    σ''   = T.mul σ' . (1.0 -) . T.diag (Diag 0) . T.matmul (T.transpose2D k')
          $ T.choleskySolve Lower k' l'

fitGPR :: Tensor -> Tensor -> GPRState -> IO (Tensor -> (Tensor, Tensor))
fitGPR x' y' gpr' = do
    θ' <- T.makeIndependentWithRequiresGrad (T.zeros [features] opts) True
    let gpr = gpr' { x  = x'
                   , y  = y'
                   , k  = T.zeros [samples, samples] opts
                   , l  = T.zeros [samples, samples] opts
                   , σ  = T.ones  [1, 1] opts
                   , μ  = T.zeros [1, 1] opts
                   , θ  = θ'
                   , ub = T.repeat [features] $ ub gpr'
                   , lb = T.repeat [features] $ lb gpr' }
    (θ'', GPRState{..}) <- runStateT fit gpr
    let x'' = T.toDevice cpu x'
        y'' = T.toDevice cpu y'
        μ'' = T.toDevice cpu μ
        σ'' = T.toDevice cpu σ
        l'' = T.toDevice cpu l
    pure $ predict' x'' y'' θ'' μ'' σ'' l''
  where
    [samples,features] = T.shape x'

mkGPR :: Int -> Double -> Double -> Double -> GPRState
mkGPR num lr' ub' lb' = GPRState { nr = num
                                 , op = opt
                                 , α  = α'
                                 , x  = T.zeros [1,1] opts
                                 , y  = T.zeros [1,1] opts
                                 , k  = T.zeros [1,1] opts
                                 , l  = T.zeros [1,1] opts
                                 , σ  = T.ones  [1,1] opts
                                 , μ  = T.zeros [1,1] opts
                                 , θ  = unsafePerformIO . T.makeIndependent
                                      $ T.zeros [1,1] opts
                                 , ub = T.asTensor' ub' opts
                                 , lb = T.asTensor' lb' opts }
  where
    α'  = T.asTensor' lr' opts
    opt = def { T.adamLr          = lr'
              , T.adamBetas       = (0.9, 0.999)
              , T.adamEps         = 1.0e-6
              , T.adamWeightDecay = 0
              , T.adamAmsgrad     = False
              } :: T.AdamOptions

trainModel :: Int -> IO ()
trainModel num = do
    modelDir <- createModelDir "./models"

    -- (header,datRaw) <- readTSV dataPath
    let (header,datRaw) = mkData 100 5 10

    let nRows = head $ T.shape datRaw
    idx <- T.multinomialIO' (T.arange' 0 nRows 1) n 

    let trainX' = headerSelect header paramsX datRaw
        trainY' = headerSelect header paramsY datRaw
        minX    = fst . T.minDim (Dim 0) RemoveDim $ trainX'
        maxX    = fst . T.maxDim (Dim 0) RemoveDim $ trainX'
        minY    = fst . T.minDim (Dim 0) RemoveDim $ trainY'
        maxY    = fst . T.maxDim (Dim 0) RemoveDim $ trainY'
        trainX  = flip T.withTensorOptions opts . scale minX maxX
                $ T.indexSelect 0 idx trainX'
        trainY  = flip T.withTensorOptions opts . scale minY maxY
                $ T.indexSelect 0 idx trainY'

    gpr <- fitGPR trainX trainY $ mkGPR num 1.0e-3 1.0 0.0

    let predictor x = T.hstack . map (T.reshape [-1,1]) $ [m,s]
          where
            (m',s) = gpr $ scale minX maxX x
            m      = scale' minY maxY m'

    idx' <- T.multinomialIO' (T.arange' 0 nRows 1) 10 
    let testX = T.indexSelect 0 idx' $ headerSelect header paramsX datRaw
        testY = T.indexSelect 0 idx' $ headerSelect header paramsY datRaw
        predY = predictor testX

    print testY
    print predY
    print . T.abs . flip T.div testY $ T.sub testY predY

    -- let (_,testD) = mkData 100 2 2
    --     testX = headerSelect header paramsX testD
    --     testY = headerSelect header paramsY testD

    --GPR.traceModel predictor >>= GPR.saveInferenceModel modelDir
    --mdl <- unTraceModel <$> loadInferenceModel modelDir 

    -- testModel paramsX paramsY predictor testX testY

    pure ()
  where
    dataPath = "./data/volumes.txt"
    n        = 200
    paramsX  = ["r_th", "g_th"]
    paramsY  = ["volume"]
    -- maskX    = boolMask' ["r_th"] paramsX
    -- maskY    = boolMask' ["volume"] paramsY

testModel :: [String] -> [String] -> (Tensor -> Tensor) -> Tensor -> Tensor -> IO ()
testModel paramsX paramsY mdl xs ys = do
    print ys
    print ys'
    print . T.abs . flip T.div ys $ T.sub ys ys'
    linePlot "Volume in cm^3" "R_th in Ohm" ["tru", "prd"] xs $ T.hstack [ys, ys']
    compPlot "Volume in cm^3" ys ys'

    pure ()
  where
    ys'' = mdl . flip T.withTensorOptions opts $ xs
    ys'  = T.reshape [-1,1] $ T.select 1 0 ys''

mkData :: Int -> Int -> Int -> ([String], Tensor)
mkData n l u = (header,values)
  where
    ds     = [l .. u]
    d      = length ds
    xs'    = T.reshape [-1,1] $ T.linspace' @Float @Float 0.0 1.0 n
    xs     = T.repeat [d, 1] xs'
    ys     = T.vstack $ map (T.exp . T.negative . flip T.mulScalar xs') ds
    zs     = T.vstack [ T.mulScalar b $ T.ones' [n,1] | b <- ds]
    values = T.hstack [xs,zs,ys]
    header = ["r_th","g_th","volume"]

traceModel :: (Tensor -> (Tensor,Tensor)) -> IO ScriptModule
traceModel p = do
    !rm <- T.trace "GaN" "forward" fun [x]
    T.toScriptModule rm
  where
    fun [x'] = let (m,s) = p x'
                in pure [T.hstack $ map (T.reshape [-1,1]) [m,s]]
    r = T.linspace' @Float @Float 0.0 1.0 10
    g = T.linspace' @Float @Float 5.0 11.0 3
    x = T.cartesian_prod [r,g]

-- | Trace to Function
unTraceModel :: ScriptModule -> (Tensor -> Tensor)
unTraceModel model' x = y
  where
    T.IVTensor y = T.runMethod1 model' "forward" $ T.IVTensor x

-- | Save a Traced ScriptModule
saveInferenceModel :: FilePath -> ScriptModule -> IO ()
saveInferenceModel path model = T.saveScript model
                              $ path ++  "/trace.pt"

-- | Load a Traced ScriptModule
loadInferenceModel :: FilePath -> IO ScriptModule
loadInferenceModel path = T.loadScript T.WithoutRequiredGrad
                        $ path ++ "/trace.pt"
