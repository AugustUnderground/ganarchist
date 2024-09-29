{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE ExtendedDefaultRules #-}

-- | Utilies for Plotting
module Plot where

import           Torch                            ( Tensor )
import qualified Torch                     as T
import qualified Torch.Functional.Internal as T   (negative)
import           Graphics.Matplotlib              ( (%), (@@), (#), (##), o1, o2
                                                  , Matplotlib)
import qualified Graphics.Matplotlib       as Plt

import           Lib

toList :: Tensor -> [Double]
toList = T.asValue @[Double] . T.toDType T.Double . T.squeezeAll

linePlot :: String -> String -> [String] -> Tensor -> Tensor -> IO ()
linePlot xlabel ylabel labels xs ys = Plt.onscreen plt
  where
    xs' = toList xs
    ys' = map toList $ cols ys
    plt = Plt.plot xs' ys' % Plt.xlabel xlabel
                           % Plt.ylabel ylabel
                           % Plt.grid True
                           @@ [o2 "labels" labels]

compPlot :: String -> Tensor -> Tensor -> IO ()
compPlot label xs ys = Plt.onscreen plt
  where
    xs'    = toList xs
    ys'    = toList ys
    xlabel = "Truth: " ++ label
    ylabel = "Prediction: " ++ label
    plt    = Plt.scatter xs' ys' % Plt.xlabel xlabel
                                 % Plt.ylabel ylabel
                                 % Plt.grid True
                                 @@ []

scatterPlot :: String -> String -> [String] -> Tensor -> Tensor -> IO ()
scatterPlot xlabel ylabel labels xs ys = Plt.onscreen plt
  where
    xs' = toList xs
    ys' = map toList $ cols ys
    plt = Plt.scatter xs' ys' % Plt.xlabel xlabel
                              % Plt.ylabel ylabel
                              % Plt.grid True
                              @@ [o2 "labels" labels]

foo :: IO ()
foo = do
    let xs = T.reshape [-1,1] $ T.linspace' @Float @Float 0.0 1.0 100
        zs = T.reshape [1,-1] $ T.arange' 5 11 1
        ys = T.exp . T.negative $ T.matmul xs zs
        xs' = toList xs
        ys' = map toList $ rows ys
        plt = Plt.plot xs' ys' % Plt.xlabel "x" % Plt.ylabel "y" % Plt.grid True

    Plt.onscreen plt
    pure ()
