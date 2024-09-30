{-# OPTIONS_GHC -Wall #-}

-- | Hyper Parameters of Net
module HyperParameters where

import           Data.Default.Class                 (Default (def))
import           Torch                              (Tensor)
import qualified Torch                         as T
import qualified Torch.Functional.Internal     as T (mish)
import qualified Torch.Optim.CppOptim          as T

-- | Activation Function is Torch.mish
φ :: Tensor -> Tensor
φ = T.mish

-- | Learning Rate
α :: Double
α = 1.0e-4

-- | Learning Rate as Torch.Tensor
α' :: Tensor
α' = T.asTensor α

-- | Estimated Moment Decay
β1 :: Double
β1 = 0.900

-- | Estimated Moment Decay
β2 :: Double
β2 = 0.999

-- | Batch Size
bs' :: Int
bs' = 4

-- | Type Alias for Adam C++ Optimizer
type Optim = T.CppOptimizerState T.AdamOptions

-- | Default Optimizer Options
opt' :: T.AdamOptions
opt' = def { T.adamLr          = α
           , T.adamBetas       = (β1, β2)
           , T.adamEps         = 1.0e-8
           , T.adamWeightDecay = 0.0
           , T.adamAmsgrad     = False }
