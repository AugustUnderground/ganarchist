{-# OPTIONS_GHC -Wall #-}

module Main (main) where

-- import Run (trainModel)
import GPR (trainModel)

main :: IO ()
main = do 
    putStrLn $ "Training for " ++ show epochs ++ " epochs"
    trainModel epochs
  where
    epochs = 12
