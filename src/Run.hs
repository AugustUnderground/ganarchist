{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE MultiParamTypeClasses #-}

-- | Run Model Training 
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
    liftIO . putStrLn $ "\tValid Loss: " ++ show vl
    put s { lossBuffer' = [], validLoss = vl : validLoss }
validEpoch (b:bs) = do
    s@TrainState{..} <- get
    let l = validStep model b
    put s {lossBuffer' = l : lossBuffer'}
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
        tl = T.asValue l'
    liftIO . putStrLn $ "\tTrain Loss: " ++ show tl
    put s {lossBuffer = [], trainLoss = tl : trainLoss}
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
    idx <- liftIO . flip T.multinomialIO' nRows . T.toDevice gpu $ T.arange' 0 nRows 1
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
        liftIO $ saveCheckPoint modelPath model optim
    let epoch' = epoch - 1
    put $ s {epoch = epoch'}
    if epoch' <= 0 then pure model else runTraining td vd

-- | Main Training Function
train :: Int -> IO ()
train num = do
    modelDir <- createModelDir "./models"

    (header,datRaw) <- loadCSVs dataPath

    let idxHi = fromJust $ elemIndex "energy_high" header
        idxLo = fromJust $ elemIndex "energy_low" header
        hi    = T.select 1 idxHi datRaw
        lo    = T.select 1 idxLo datRaw
        msk   = T.logicalAnd (T.lt hi 500.0e-6 `T.logicalAnd` T.gt hi 10.0e-9)
                             (T.lt lo 500.0e-6 `T.logicalAnd` T.gt lo 10.0e-9)
        dat'  = maskSelect 0 msk datRaw
        nRows = head $ T.shape dat'

    !datShuffled <- flip (T.indexSelect 0) dat' <$> T.multinomialIO' (T.arange' 0 nRows 1) nRows 

    let ts    = floor @Float $ 0.85 * realToFrac (head $ T.shape datShuffled)
        datX' = headerSelect header paramsX datShuffled
        datY' = headerSelect header paramsY datShuffled
        minX  = fst . T.minDim (Dim 0) RemoveDim $ datX'
        maxX  = fst . T.maxDim (Dim 0) RemoveDim $ datX'
        minY  = fst . T.minDim (Dim 0) RemoveDim $ datY'
        maxY  = fst . T.maxDim (Dim 0) RemoveDim $ datY'
        datX  = scale minX maxX datX'
        datY  = scale minY maxY datY'
        dat   = T.toDevice gpu $ T.cat (Dim 1) [datX, datY]

    let [datTrain,datValid] = T.split ts (Dim 0) dat

    net <- T.toDevice gpu <$> T.sample spec
    opt <- T.initOptimizer opt' $ T.flattenParameters net

    let initialState = TrainState num l l l l net opt α' bs' modelDir paramsX paramsY

    -- evalStateT (runTraining datTrain datValid) initialState
    --     >>= withGrad >>= saveCheckPoint modelDir
    _ <- runStateT (runTraining datTrain datValid) initialState

    -- let modelDir = "models/20240914-183529"
    !net' <- loadCheckPoint modelDir spec >>= noGrad . fst
    let predict = scale' minY maxY . forward net' . scale minX maxX

    traceModel dimX paramsX paramsY predict >>= saveInferenceModel modelDir 
    traceGraph dimX predict >>= saveONNXModel modelDir

    !net'' <- loadInferenceModel modelDir >>= noGrad . unTraceModel 

    testModel paramsY net'' datX' datY'

    putStrLn $ "Final checkpoint in " ++ modelDir
  where
    l        = []
    dataPath = "./data"
    paramsY  = ["C_ISS", "C_OSS", "C_RSS", "I_DSS", "Q_G", "Q_GD", "Q_GS", "R_DS_on", "R_G"]
    paramsX  = ["iload_max"]
    dimY     = length paramsY
    dimX     = length paramsX
    spec     = NetSpec dimX dimY

testModel :: [String] -> (T.Tensor -> T.Tensor) -> Tensor -> Tensor -> IO ()
testModel paramsY net xs ys = do
    let ys'  = net xs
        mape = T.asValue @[Float] . T.meanDim (Dim 0) RemoveDim T.Float
             . T.mulScalar @Float 100.0 . T.abs $ (ys - ys') / ys
    putStrLn "Prediction MAPEs"
    mapM_ putStrLn $ zipWith (\p m -> p ++ ":\t" ++ show m ++ "%") paramsY mape
