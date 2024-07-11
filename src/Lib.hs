{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Lib ( liftState, toStatic, splitBatches, readCSV
           ) where

import           Control.Monad.State
import           Torch.Typed               (Tensor (..))
import qualified Torch.Typed         as T
import qualified Torch               as UT
import           Data.List.Split           (splitOn)

-- | Lift State into StateT
liftState :: forall m s a. (Monad m) => State s a -> StateT s m a
liftState s = do
   state1 <- get
   ( let (result', state') = evalState (do { result'' <- s
                                           ; state'' <- get
                                           ; return (result'', state'')
                                           }) state1
      in do
           put state'
           return result' )

-- | Dynamic to static tensor
toStatic :: forall r c d. UT.Tensor -> Tensor d 'T.Float '[r, c]
toStatic = UnsafeMkTensor @d @'T.Float @'[r, c] 

-- | Split Tensor into List of GPU Tensors
splitBatches :: Int -> UT.Tensor -> [UT.Tensor]
splitBatches bs = filter ((bs==) . head . UT.shape) . UT.split bs (UT.Dim 0)
                . UT.toDevice (T.Device T.CUDA 1)

-- | Read Delimited data into Tensor
readDSV :: String -> FilePath -> IO ([String], UT.Tensor)
readDSV sep path = do
    file <- lines <$> readFile path
    let col = splitOn sep $ head file
        dat = UT.asTensor . map (map (read @Float) . splitOn sep) $ tail file
    pure (col, dat)

readCSV :: FilePath -> IO ([String], UT.Tensor)
readCSV = readDSV ","
