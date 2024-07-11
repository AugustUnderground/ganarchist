{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE StrictData #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE IncoherentInstances #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PartialTypeSignatures #-}

module Net where

import           Torch.Typed                   ( Tensor, DType, DeviceType
                                               , Linear, Parameterized, HasForward
                                               , StandardFloatingPointDTypeValidation
                                               , KnownDType, KnownDevice, All
                                               , RandDTypeIsValid, Randomizable
                                               , LinearSpec (..), DeviceType (..)
                                               , Adam, Reduction (..), HasGrad
                                               , HMap', HList, Parameters, ToDependent
                                               , HMapM', MakeIndependent
                                               )
import qualified Torch.Typed                   as T
import           Torch.Internal.Class          (Castable)
import           Torch                         (ATenTensor, KeepDim (..), Dim (..))
import qualified Torch                         as UT

-- import qualified Torch.Extensions              as UT
-- import qualified Data.Frame                    as DF
import           Lib


import           Control.Monad                 (when)
import           Control.Monad.State
import           Data.List                     (isSuffixOf)
import           GHC.TypeLits
import           GHC.Generics

-- | GPU
gpu :: UT.Device
gpu = UT.Device UT.CUDA 0

-- | NN Specification
data NetSpec (xDim :: Nat) (yDim :: Nat)
             (dtype :: DType) (device :: (DeviceType, Nat)) = NetSpec
    deriving (Eq, Show)

-- | NN
data Net (xDim :: Nat) (yDim :: Nat)
         (dtype :: DType) (device :: (DeviceType, Nat)) =
        Net { l0 :: !(Linear xDim 32   dtype device)
            , l1 :: !(Linear 32   128  dtype device)
            , l2 :: !(Linear 128  512  dtype device)
            , l3 :: !(Linear 512  128  dtype device)
            , l4 :: !(Linear 128  64   dtype device)
            , l5 :: !(Linear 64   32   dtype device)
            , l6 :: !(Linear 32   yDim dtype device)
            } deriving (Show, Generic, Parameterized)

instance (StandardFloatingPointDTypeValidation device dtype) => HasForward
    (Net xDim yDim dtype device) (Tensor device dtype '[n, xDim])
                                 (Tensor device dtype '[n, yDim])
  where
    forward Net{..} = T.forward l6 . T.relu
                    . T.forward l5 . T.relu
                    . T.forward l4 . T.relu
                    . T.forward l3 . T.relu
                    . T.forward l2 . T.relu
                    . T.forward l1 . T.relu
                    . T.forward l0
    forwardStoch = (pure .) . T.forward

instance ( KnownDevice device
         , KnownDType dtype
         , All KnownNat '[xDim, yDim]
         , RandDTypeIsValid device dtype
         ) => Randomizable (NetSpec xDim yDim dtype device) (Net xDim yDim dtype device)
  where
    sample NetSpec = Net <$> T.sample LinearSpec
                         <*> T.sample LinearSpec 
                         <*> T.sample LinearSpec 
                         <*> T.sample LinearSpec 
                         <*> T.sample LinearSpec 
                         <*> T.sample LinearSpec 
                         <*> T.sample LinearSpec 

-- | Training State
data TrainState = TrainState { epoch        :: !Int          -- ^ Current Epoch
                             , trainLoss    :: !Loss         -- ^ Training Loss
                             , validLoss    :: !Loss         -- ^ Validation Loss
                             , model        :: !Model        -- ^ NN
                             , optim        :: !Optim        -- ^ Optimizier
                             , learningRate :: !LearningRate -- ^ Learning Rate
                             , modelPath    :: !FilePath     -- ^ Save Path
                             }

-- | Data Batch
data Batch = Batch { xs :: !(Tensor TrainDevice 'T.Float '[BatchSize, XDim]) -- ^ Inputs
                   , ys :: !(Tensor TrainDevice 'T.Float '[BatchSize, YDim]) -- ^ Outputs
                   } deriving (Show)

-- | Training Device
type TrainDevice  = '( 'CUDA, 1)
-- | Input Dimension
type XDim         = 10
-- | Output Dimension
type YDim         = 2
-- | Batchsize
type BatchSize    = 24
-- | Loss Type
type Loss         = Tensor TrainDevice 'T.Float '[]
-- | Learning Rate
type LearningRate = T.LearningRate TrainDevice 'T.Float
-- | NN Module
type Model        = Net XDim YDim 'T.Float TrainDevice
-- | Optimizer
type Optim        = (Adam '[ Tensor TrainDevice 'T.Float '[32, XDim]
                           , Tensor TrainDevice 'T.Float '[32]
                           , Tensor TrainDevice 'T.Float '[128, 32]
                           , Tensor TrainDevice 'T.Float '[128]
                           , Tensor TrainDevice 'T.Float '[512, 128]
                           , Tensor TrainDevice 'T.Float '[512]
                           , Tensor TrainDevice 'T.Float '[128, 512]
                           , Tensor TrainDevice 'T.Float '[128]
                           , Tensor TrainDevice 'T.Float '[64, 128]
                           , Tensor TrainDevice 'T.Float '[64]
                           , Tensor TrainDevice 'T.Float '[32, 64]
                           , Tensor TrainDevice 'T.Float '[32]
                           , Tensor TrainDevice 'T.Float '[YDim, 32]
                           , Tensor TrainDevice 'T.Float '[YDim] ])

-- | Validation Step without gradient
validStep :: ( HasForward f (Tensor TrainDevice 'T.Float '[BatchSize, XDim])
                            (Tensor TrainDevice 'T.Float '[BatchSize, YDim])
             ) => f -> Batch -> Loss
validStep m Batch{..} = T.l1Loss @'ReduceMean ys' ys
  where
    ys' = T.forward m xs

-- | Validation Epoch
validEpoch :: forall model.
    ( (model ~ Model)
    , HasForward model (Tensor TrainDevice 'UT.Float '[BatchSize, XDim])
                       (Tensor TrainDevice 'UT.Float '[BatchSize, YDim])
    ) => [Batch] -> State TrainState Loss
validEpoch [] = gets validLoss
validEpoch (b:bs) = do
    s@TrainState{..} <- get
    let validLoss' = validStep model b
    put $ s {validLoss = validLoss'}
    validEpoch bs 

-- | Training Step with Gradient
trainStep :: ( HasForward a (Tensor TrainDevice 'UT.Float '[BatchSize, XDim])
                            (Tensor TrainDevice 'UT.Float '[BatchSize, YDim])
             , Parameterized a
             , HasGrad (HList (Parameters a)) (HList gradients)
             , HMap' ToDependent (Parameters a) gradients
             , Castable (HList gradients) [ATenTensor]
             , T.Optimizer b gradients gradients 'UT.Float TrainDevice
             , HMapM' IO MakeIndependent gradients (Parameters a)
             ) => T.LearningRate TrainDevice 'UT.Float
               -> a -> b -> Batch -> IO (a, b, Loss)
trainStep α m o Batch{..} = do
    let ys' = T.forward m xs
        -- l'  = T.smoothL1Loss @'T.ReduceMean ys' ys
        l' = T.l1Loss @'ReduceMean ys' ys
    (m', o') <- T.runStep m o l' α
    pure (m', o', l')

-- | Training Epoch
trainEpoch :: forall model.
    ( (model ~ Model)
    , T.HasForward model (Tensor TrainDevice 'UT.Float '[BatchSize, XDim])
                         (Tensor TrainDevice 'UT.Float '[BatchSize, YDim])
    ) => [Batch] -> StateT TrainState IO Loss
trainEpoch   []   = gets trainLoss
trainEpoch (b:bs) = do
    s@TrainState{..} <- get
    (m', o', l') <- liftIO $ trainStep learningRate model optim b
    put $ s { model = m', optim = o', trainLoss = l' }
    trainEpoch bs

-- | Training in State Monad
runTraining :: forall model.
    ( (model ~ Model)
    , HasForward model (Tensor TrainDevice 'UT.Float '[BatchSize, XDim])
                       (Tensor TrainDevice 'UT.Float '[BatchSize, YDim])
    ) => [Batch] -> [Batch] -> StateT TrainState IO Model
runTraining tb vb = do
    tl <- trainEpoch tb
    vl <- liftState $ validEpoch vb
    s@TrainState{..} <- get
    when (UT.asValue @Bool . T.toDynamic $ T.lt vl validLoss) $ do
        liftIO . flip T.save (modelPath ++ ".pt") . T.hmap' T.ToDependent
               . T.flattenParameters $ model
    liftIO . putStrLn $ "Epoch " ++ show epoch ++ ":"
    liftIO . putStrLn $ "\tTrain Loss: " ++ show tl
    liftIO . putStrLn $ "\tValid Loss: " ++ show vl
    let epoch' = epoch - 1
    put $ s {epoch = epoch'}
    if epoch' <= 0
       then pure model
       else runTraining tb vb

-- | Main Training Function
-- train :: FilePath -> FilePath -> IO ()
-- train dataPath modelPath = do
--     (col,dat) <- readCSV ";" dataPath
-- 
--     let -- cols    = DF.columns df'
--         -- paramsX = filter (\c -> isSuffixOf "1_gmoverid" c || isSuffixOf "1_fug" c) cols
--         --             ++ ["MNCM12_id", "MPCM22_id", "MPCM32_id"]
--         maskX'  = UT.toDevice gpu
--                 $ UT.boolMask' (filter (isSuffixOf "_fug") paramsX) paramsX
--         maskY   = UT.toDevice UT.cpu
--                 $ UT.boolMask' ["sr_r", "ugbw"] paramsY
--         dfT     = DF.dropNan
--                 $ DF.union (UT.trafo maskX <$> DF.lookup paramsX df')
--                            (UT.trafo maskY <$> DF.lookup paramsY df')
--         dfX'    = DF.lookup paramsX dfT
--         dfY'    = DF.lookup paramsY dfT
--         minX    = fst . UT.minDim (Dim 0) RemoveDim . DF.values $ dfX'
--         maxX    = fst . UT.maxDim (Dim 0) RemoveDim . DF.values $ dfX'
--         minY    = fst . UT.minDim (Dim 0) RemoveDim . DF.values $ dfY'
--         maxY    = fst . UT.maxDim (Dim 0) RemoveDim . DF.values $ dfY'
--         !dfX    = UT.scale minX maxX <$> dfX'
--         !dfY    = UT.scale minY maxY <$> dfY'
--         !df     = DF.dropNan $ DF.union dfX dfY
-- 
--     (trainX', validX', trainY', validY') <- DF.trainTestSplit paramsX paramsY r 
--                                             <$> DF.sampleIO np False df
-- 
--     let trainX  = map toStatic . splitBatches bs
--                 $ trainX' :: [Tensor TrainDevice 'T.Float '[BatchSize, XDim]]
--         trainY  = map toStatic . splitBatches bs
--                 $ trainY' :: [Tensor TrainDevice 'T.Float '[BatchSize, YDim]]
--         validX  = map toStatic . splitBatches bs
--                 $ validX' :: [Tensor TrainDevice 'T.Float '[BatchSize, XDim]]
--         validY  = map toStatic . splitBatches bs
--                 $ validY' :: [Tensor TrainDevice 'T.Float '[BatchSize, YDim]]
--         trainBs = zipWith Batch trainX trainY
--         validBs = zipWith Batch validX validY
-- 
--     mdl <- T.sample (NetSpec :: NetSpec XDim YDim 'T.Float TrainDevice)
--     let opt = T.mkAdam 0 β1 β2 $ T.flattenParameters mdl
-- 
--     let initialState = TrainState e l l mdl opt α "./models/foobar"
-- 
--     -- finalState <- execStateT (runTraining trainBs validBs) initialState
--     mdl' <- evalStateT (runTraining trainBs validBs) initialState
-- 
--     flip T.save (modelPath ++ "-typed.pt")
--         . T.hmap' T.ToDependent
--         . T.flattenParameters
--         $ T.toDevice @'( 'T.CPU, 0) @TrainDevice mdl'
-- 
--     -- mdl'' <- traceAndSave modelPath minX maxX minY maxY maskX maskY mdl'
--     -- testModel mdl'' df' paramsX paramsY 
--     pure ()
--   where
--     l  = 1.0 / (T.zeros :: T.Tensor TrainDevice 'T.Float '[])
--     α  = 1.0e-3 :: LearningRate
--     β1 = 0.9 :: Float
--     β2 = 0.999 :: Float
--     bs = 24
--     np = 10000
--     r  = 0.7
--     e  = 6
--     paramsY = ["energy_high", "energy_low"]
--     paramsX = [ "iload_ratio", "rgoff", "rgon", "t_dead", "vsup_ratio"
--               , "I_DSS", "R_DS_on", "R_G", "C_ISS", "C_OSS", "C_RSS"
--               , "Q_G", "Q_GD", "Q_GS"]
--     maskX   = ["I_DSS", "R_DS_on", "R_G", "C_ISS", "C_OSS", "C_RSS", "Q_G", "Q_GD", "Q_GS"]
