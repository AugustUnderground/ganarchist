{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE MultiParamTypeClasses #-}

-- | Statically Typed Model Training
module Run where

import           Lib
import           Net
import           HyperParameters
import           Data.List                 (elemIndex)
import           Data.Maybe                (fromJust)
import           Control.Monad             (when)
import           Control.Monad.State       (gets, MonadIO (..), MonadState (..), StateT (..))
import           Torch                     ( Tensor, Dim (..), LearningRate
                                           , KeepDim (..), Reduction (..))
import qualified Torch                as T
import qualified Torch.Optim.CppOptim as T

-- | A single batch of Data
data Batch = Batch { xs :: !Tensor -- ^ Input Data
                   , ys :: !Tensor -- ^ Output Data
                   } deriving (Show, Eq)

-- | Training State
data TrainState = TrainState { epoch        :: !Int          -- ^ Current Epoch
                             , lossBuffer   :: ![Tensor]     -- ^ Buffer for batch losses
                             , lossBuffer'  :: ![Tensor]     -- ^ Buffer for batch losses
                             , trainLoss    :: ![Float]      -- ^ Training Loss
                             , validLoss    :: ![Float]      -- ^ Validation Loss
                             , model        :: !Net          -- ^ Surrogate Model
                             , optim        :: !Optim        -- ^ Optimizier
                             , learningRate :: !LearningRate -- ^ Learning Rate
                             , batchSize    :: !Int          -- ^ Batch Size
                             , modelPath    :: !FilePath     -- ^ Save Path
                             , xParams      :: ![String]     -- ^ X Columns
                             , yParams      :: ![String]     -- ^ Y Columns
                             }

-- | Validation Step without gradient
validStep :: Net -> Batch -> Tensor
validStep net Batch{..} = T.meanDim (Dim 0) RemoveDim T.Float
                        $ T.l1Loss ReduceNone ys' ys
  where
    ys' = forward net xs

-- | Validation Epoch
validEpoch :: [Batch] -> StateT TrainState IO ()
validEpoch [] = do 
    s@TrainState{..} <- get
    let l' = T.stack (Dim 0) lossBuffer'
        l  = T.meanDim (Dim 0) RemoveDim T.Float l'
        vl = T.asValue $ T.mean l
    liftIO . putStrLn $ "\tValid Loss: " ++ show l
    put s { lossBuffer' = [], validLoss = vl : validLoss }
validEpoch (b:bs) = do
    s@TrainState{..} <- get
    let l = validStep model b
    put $ s {lossBuffer' = l : lossBuffer'}
    validEpoch bs 

-- | Training Step with Gradient
trainStep ::  LearningRate -> Net -> Optim -> Batch
          -> IO (Net, Optim, T.Tensor)
trainStep a m o Batch{..} = do
    (m', o') <- T.runStep m o l a
    pure (m', o', l)
  where
    l = T.l1Loss T.ReduceSum ys $ forward m xs

-- | Training Epoch
trainEpoch :: [Batch] -> StateT TrainState IO ()
trainEpoch [] = do
    s@TrainState{..} <- get
    let l' = T.mean . T.cat (Dim 0) . map (T.reshape [1]) $ lossBuffer
        l  = T.asValue l'
    liftIO . putStrLn $ "\tTrain Loss: " ++ show l'
    put s {lossBuffer = [], trainLoss = l : trainLoss}
    pure ()
trainEpoch (b:bs) = do
    s@TrainState{..} <- get
    (model', optim', loss') <- liftIO $ trainStep learningRate model optim b
    put $ s { model = model', optim = optim', lossBuffer = loss' : lossBuffer }
    trainEpoch bs

-- | Data Shuffler
shuffleData :: Tensor -> Tensor -> StateT TrainState IO [Batch]
shuffleData xs ys = do
    TrainState{..} <- get
    idx <- liftIO . flip T.multinomialIO' nRows $ T.arange' 0 nRows 1
    let xs' = splitBatches batchSize $ T.indexSelect 0 idx xs
        ys' = splitBatches batchSize $ T.indexSelect 0 idx ys
    pure $ zipWith Batch xs' ys'
  where
    nRows = head $ T.shape xs

-- | Training in State Monad
runTraining :: Tensor -> Tensor -> StateT TrainState IO Net
runTraining td vd = do
    ep <- gets epoch
    liftIO . putStrLn $ "Epoch " ++ show ep ++ ":"
    pX <- gets xParams
    pY <- gets yParams
    let header = pX ++ pY
        xs     = headerSelect header pX td
        ys     = headerSelect header pY td
        xs'    = headerSelect header pX vd
        ys'    = headerSelect header pY vd
    shuffleData xs  ys  >>= trainEpoch
    shuffleData xs' ys' >>= validEpoch
    s@TrainState{..} <- get
    when (head validLoss == minimum validLoss) $ do
        liftIO $ putStrLn "\tNew model Saved!"
        liftIO $ saveCheckPoint (modelPath ++ "-checkpoint.pt") model optim
    let epoch' = epoch - 1
    put $ s {epoch = epoch'}
    if epoch' <= 0 then pure model else runTraining td vd

-- | Main Training Function
train :: Int -> IO ()
train num = do
    modelDir <- createModelDir "./models"

    (header,datRaw) <- loadCSVs dataPath
    let nRows = head $ T.shape datRaw

    let idxHi = fromJust $ elemIndex "energy_high" header
        idxLo = fromJust $ elemIndex "energy_lo" header
        hi    = T.select 1 idxHi datRaw
        lo    = T.select 1 idxLo datRaw
        msk   =  T.logicalAnd (T.lt hi 500.0e-6 `T.logicalAnd` T.gt hi 10.0e-9)
                              (T.lt lo 500.0e-6 `T.logicalAnd` T.gt lo 10.0e-9)
        dat'  = maskSelect 0 msk datRaw

    datShuffled <- flip (T.indexSelect 0) dat' <$> T.multinomialIO' (T.arange' 0 nRows 1) nRows 

    let ts    = floor @Float $ 0.85 * realToFrac (head $ T.shape datShuffled)
        datX' = headerSelect header paramsX datShuffled
        datY' = headerSelect header paramsY datShuffled
        minX = fst . T.minDim (Dim 0) RemoveDim $ datX'
        maxX = fst . T.maxDim (Dim 0) RemoveDim $ datX'
        minY = fst . T.minDim (Dim 0) RemoveDim $ datY'
        maxY = fst . T.maxDim (Dim 0) RemoveDim $ datY'
        dfX  = scale minX maxX datX'
        dfY  = scale minY maxY datY'
        dat  = T.cat (Dim 1) [dfX, dfY]

    let [datTrain,datValid] = T.split ts (Dim 0) dat

    mdl <- T.toDevice gpu <$> T.sample (NetSpec dimX dimY)
    opt <- T.initOptimizer opt' $ T.flattenParameters mdl

    let initialState = TrainState num l l l l mdl opt α' bs' modelDir paramsX paramsY

    -- evalStateT (runTraining datTrain datValid) initialState
    --     >>= withGrad >>= saveCheckPoint modelDir
    (net',_) <- runStateT (runTraining datTrain datValid) initialState

    let predict = scale' minY maxY . forward net' . scale minX maxX

    traceModel dimX paramsX paramsY predict >>= saveInferenceModel modelDir
    net'' <- unTraceModel <$> loadInferenceModel modelDir

    -- testModel net'' xs ys

    putStrLn $ "Traced Model saved in " ++ modelDir
  where
    l        = []
    dataPath = "./data"
    paramsY  = ["C_ISS", "C_OSS", "C_RSS", "I_DSS", "Q_G", "Q_GD", "Q_GS", "R_DS_on", "R_G"]
    paramsX  = ["iload_max"]
    dimY     = length paramsY
    dimX     = length paramsX

-- testModel :: PDK.ID -> CKT.ID -> (T.Tensor -> T.Tensor) -> [String] -> [String]
--           -> DF.DataFrame T.Tensor -> IO ()
-- testModel pdk ckt mdl paramsX paramsY df = do
--     createDirectoryIfMissing True plotPath
--     dat <- DF.sampleIO 1000 False df
--     -- dat <- DF.shuffleIO df
--     let xs  = DF.values $ DF.lookup paramsX dat
--         ys  = DF.values $ DF.lookup paramsY dat
--         ys' = mdl xs
--     mapM_ (uncurry' (plt plotPath))  $ zip3 paramsY (T.cols ys) (T.cols ys') 
--     mapM_ (uncurry  (hst plotPath))  $ zip  paramsY (T.cols ys)
--     mapM_ (uncurry  (hst' plotPath)) $ zip  paramsX (T.cols xs)
--     dat' <- DF.sampleIO 10 False df
--     let x  = DF.values $ DF.lookup paramsX dat'
--         y  = DF.values $ DF.lookup paramsY dat'
--         y' = DF.values . DF.dropNan . DF.DataFrame paramsY $ mdl x
--     print x
--     print y
--     print y'
--     dat'' <- DF.sampleIO 100 False df
--     let ex = DF.values $ DF.lookup paramsX dat''
--         ey = DF.values $ DF.lookup paramsY dat''
--         ey' = DF.values . DF.dropNan . DF.DataFrame paramsY $ mdl ex
--         ae = (/ ey) $ T.l1Loss T.ReduceNone ey ey'
--         mae = T.meanDim (T.Dim 0) T.RemoveDim T.Float ae
--         mape = 100 * mae
--         mape' = T.mean mape
--     print mae
--     print mape
--     print mape'
--   where
--     plotPath = "./plots/" ++ show pdk ++ "/" ++ show ckt
