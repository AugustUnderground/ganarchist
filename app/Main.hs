{-# OPTIONS_GHC -Wall #-}

module Main (main) where

import Run (trainModel)
-- import GPR (trainModel)

main :: IO ()
main = do 
    putStrLn $ "Training Net with " ++ p ++ " for " ++ show e ++ " epochs"
    trainModel p e px py mx my
  where
    e  = 10
    p  = "./data/volumes.csv"
    px = ["rth"]
    py = ["wb", "lb", "hb", "wfin", "hfin", "nfin", "vol"]
    mx = ["rth"]
    my = ["vol"]
    -- p  = "./data/gans.csv"
    -- px = ["v_ds_work", "i_d_max"]
    -- py = ["r_ds_on","r_g","g_fs","v_gs_work","v_gs_max","v_th","c_iss","c_oss","c_rss"]
    -- mx = ["v_ds_work", "i_d_max"]
    -- my = ["r_ds_on","g_fs","c_iss","c_oss","c_rss"]
