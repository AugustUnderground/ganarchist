{-# OPTIONS_GHC -Wall #-}

module Main (main) where

import Run (train)

main :: IO ()
main = do 
    putStrLn $ "Training for " ++ show epochs ++ " epochs"
    train epochs
  where
    epochs = 30
