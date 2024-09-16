{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE StrictData #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MultiParamTypeClasses #-}

-- | Neural Network Definition
module Net where

import           Lib
import           HyperParameters
import           Data.List                      (singleton)
import           GHC.Generics                   (Generic)
import           Torch                          ( Tensor, Linear, LinearSpec (..)
                                                , Randomizable (..)
                                                , ScriptModule, Graph )
import qualified Torch                     as T
import qualified Torch.Functional.Internal as T (mish)
import           Torch.NN                       (Parameterized)
import qualified Torch.NN                  as NN
import qualified Torch.Optim.CppOptim      as T

-- | Neural Network Specification
data NetSpec = NetSpec { numX   :: !Int -- ^ Number of input neurons
                       , numY   :: !Int -- ^ Number of output neurons
                       } deriving (Show, Eq)

-- | Network Architecture
data Net = Net { fc0 :: !Linear
               , fc1 :: !Linear
               , fc2 :: !Linear
               , fc3 :: !Linear
               , fc4 :: !Linear
               } deriving (Generic, Show, Parameterized)

-- | Neural Network Weight initialization
instance Randomizable NetSpec Net where
    sample NetSpec{..} = Net <$> T.sample (LinearSpec numX 32)
                             <*> T.sample (LinearSpec 32   64)
                             <*> T.sample (LinearSpec 64   128)
                             <*> T.sample (LinearSpec 128  64)
                             <*> T.sample (LinearSpec 64   numY)

-- | Neural Network Forward Pass with scaled Data
forward :: Net -> Tensor -> Tensor
forward Net{..} = T.linear fc4 . T.mish
                . T.linear fc3 . T.mish
                . T.linear fc2 . T.mish
                . T.linear fc1 . T.mish
                . T.linear fc0 

-- | Remove Gradient for tracing / scripting
noGrad :: (Parameterized f) => f -> IO f
noGrad net = do
    params <- mapM ((`T.makeIndependentWithRequiresGrad` False) . detachToCPU)
            $ NN.flattenParameters net
    pure $ NN.replaceParameters net params
  where
    detachToCPU = T.toDevice cpu . T.toDependent

-- | Save Model and Optimizer Checkpoint
saveCheckPoint :: FilePath -> Net -> Optim -> IO ()
saveCheckPoint path net opt = do
    T.saveParams net (path ++ "/mdl.ckpt")
    T.saveState  opt (path ++ "/opt.ckpt")

-- | Load a Saved Model and Optimizer CheckPoint
loadCheckPoint :: FilePath -> NetSpec -> IO (Net, Optim)
loadCheckPoint path spec = do
    net <- T.sample spec >>= (`T.loadParams` (path ++ "/mdl.ckpt"))
    opt <- T.initOptimizer opt' $ T.flattenParameters net
    T.loadState opt (path ++ "/opt.ckpt")
    pure (net, opt)

-- | Trace and Return a Script Module
traceModel :: Int -> [String] -> [String] -> (Tensor -> Tensor) -> IO ScriptModule
traceModel num xs ys predict = do
    !rm <- T.randnIO' [10,num] >>= T.trace "GaN" "forward" fun . singleton
    T.define rm $ "def inputs(self,x):\n\treturn " ++ show xs ++ "\n"
    T.define rm $ "def outputs(self,x):\n\treturn " ++ show ys ++ "\n"
    T.toScriptModule rm
  where
    fun = pure . map predict

-- | Trace to Function
unTraceModel :: ScriptModule -> (Tensor -> Tensor)
unTraceModel model' x = y
  where
    T.IVTensor y = T.runMethod1 model' "forward" $ T.IVTensor x

-- | Save a Traced ScriptModule
saveInferenceModel :: FilePath -> ScriptModule -> IO ()
saveInferenceModel path model = T.saveScript model
                              $ path ++ "/trace.pt"

-- | Load a Traced ScriptModule
loadInferenceModel :: FilePath -> IO ScriptModule
loadInferenceModel path = T.loadScript T.WithoutRequiredGrad
                        $ path ++ "/trace.pt"

-- | Trace Torch Module with `num` inputs as grpah
traceGraph :: Int -> (Tensor -> Tensor) -> IO Graph
traceGraph num predict = T.randnIO' [10,num]
                          >>= T.traceAsGraph fun . singleton
  where
    fun = pure . map predict

-- | Save ONNX Model
saveONNX :: FilePath -> Int -> (Tensor -> Tensor) -> IO ()
saveONNX path num predict = traceGraph num predict
                              >>= T.printOnnx >>= writeFile path
