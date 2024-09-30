{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ScopedTypeVariables #-}

-- | Utilies for Training Transistor Models
module Lib ( liftState
           , splitBatches
           , readCSV
           , loadCSVs
           , readTSV
           , Transistor (..)
           , cpu
           , gpu
           , maskSelect
           , maskSelect'
           , headerSelect
           , boolMask'
           , scale
           , scale'
           , trafo
           , trafo'
           , createModelDir
           , mish
           , lhsRandom
           , lhsMaxMin
           , cholesky'
           , choleskySolve'
           , clamp'
           , multivariateNormalIO'
           , rows
           , cols
           ) where

import           Data.Time.Clock                (getCurrentTime)
import           Data.Time.Format               (formatTime, defaultTimeLocale)
import           System.Directory
import           Data.List                      (elemIndex, isPrefixOf, isInfixOf)
import           Data.Maybe                     (fromJust)
import           Data.List.Split                (splitOn)
import           Control.Monad                  (replicateM)
import           Control.Monad.State            ( evalState, MonadState (put, get)
                                                , State, StateT )
import           Rando                          (shuffle)
import           Torch                          ( Tensor, Device (..), Tri (..)
                                                , DeviceType (..), Dim (..)
                                                , KeepDim (..) )
import qualified Torch                     as T
import qualified Torch.Functional.Internal as T ( powScalar', pdist, linalg_cholesky
                                                , maximum, minimum )

-- | Available Transistors
data Transistor = GS66502B -- ^ GaN Systems GS66502B
                | GS66504B -- ^ GaN Systems GS66504B
                | GS66506T -- ^ GaN Systems GS66506T
                | GS66508T -- ^ GaN Systems GS66508T
                | GS66516T -- ^ GaN Systems GS66516T
    deriving (Eq, Enum, Bounded, Show, Read)

-- | Default GPU Device
gpu :: Device
gpu = Device CUDA 1

-- | Default CPU Device
cpu :: Device
cpu = Device CPU 0

-- | Inverse of log10
pow10 :: Tensor -> Tensor
pow10 = T.powScalar' 10.0

-- | Re-implemented log10
log10' :: Tensor -> Tensor
log10' x = T.div (T.log x) . T.log . T.mulScalar @Float 10.0 $ T.onesLike x

-- | Torch.indexSelect but with a boolean mask
maskSelect :: Int -> Tensor -> Tensor -> Tensor
maskSelect dim msk = T.indexSelect dim idx
  where
    idx = T.squeezeAll $ T.nonzero msk

-- | Torch.indexSelect' but with a boolean mask
maskSelect' :: Int -> [Bool] -> Tensor -> Tensor
maskSelect' dim msk' = maskSelect dim msk
  where
    msk = T.asTensor msk'

-- | Scale Torch.Tensor to [0,1] given a min and max
scale :: Tensor -- ^ min
      -> Tensor -- ^ max
      -> Tensor -- ^ un-scaled input
      -> Tensor -- ^ scaled output
scale xMin xMax x = (x - xMin) / (xMax - xMin)

-- | Un-Scale Torch.Tensor from [0,1] given a min and max
scale' :: Tensor -- ^ min
       -> Tensor -- ^ max
       -> Tensor -- ^ scaled input
       -> Tensor -- ^ un-scaled output
scale' xMin xMax x = (x * (xMax - xMin)) + xMin

-- | Apply log10 to masked data
trafo :: Tensor -> Tensor -> Tensor
trafo m x = T.add (T.mul m' x) . T.log10 . T.add m' . T.mul m . T.abs $ x
  where
    m' = T.sub (T.onesLike x) m 

-- | Apply pow10 to masked data
trafo' :: Tensor -> Tensor -> Tensor
trafo' m x = T.add (T.mul m' x) . T.mul m . pow10 $ T.mul m x
  where
    m' = T.sub (T.onesLike x) m 

-- | Mish activation function
mish :: Tensor -> Tensor
mish x = T.mul x . T.tanh . T.log . T.addScalar @Float 1.0 $ T.exp x

-- | like @clamp@ but with higher dimensional bounds
clamp' :: Tensor -> Tensor -> Tensor -> Tensor
clamp' l u = T.maximum l . T.minimum u

-- | cholesky solve for batchsize 1
choleskySolve' :: Tensor -> Tensor -> Tensor
choleskySolve' x1 x2 = T.squeezeDim 0 $ T.choleskySolve Lower x1' x2'
  where
    x1' = T.unsqueeze (Dim 0) x1
    x2' = T.unsqueeze (Dim 0) x2

-- | Linalg cholesky
cholesky' :: Tri -> Tensor -> Tensor
cholesky' Lower = flip T.linalg_cholesky False
cholesky' Upper = flip T.linalg_cholesky True

-- | Multivaraite Normal Sample
multivariateNormalIO' :: Int -> Tensor -> Tensor -> IO Tensor
multivariateNormalIO' n μ σ = do 
    z <- T.toDType T.Double <$> T.randnIO' [d,n]
    pure . T.transpose2D $ T.matmul l z + T.unsqueeze (Dim 1) μ
  where
    d = head $ T.shape μ
    l = cholesky' Lower σ

-- | Create a boolean mask from a subset of column names
boolMask :: [String] -> [String] -> [Bool]
boolMask sub = map (`elem` sub)

-- | Create a boolean mask Tensor from a subset of column names
boolMask' :: [String] -> [String] -> Tensor
boolMask' sub set = T.asTensor' (boolMask sub set) 
                  $ T.withDType T.Float T.defaultOpts

-- | Select columns based in strings in string list
headerSelect :: [String] -> [String] -> Tensor -> Tensor
headerSelect header cols = T.reshape [-1, len] . T.indexSelect' 1 idx
  where
    len = length cols
    idx = map (fromJust . flip elemIndex header) cols

-- | Convert Tensor to List of 1D vectors, column-wise
cols :: Tensor -> [Tensor]
cols = map T.squeezeAll . T.split 1 (T.Dim 1)

-- | Convert Tensor to List of 1D vectors, row-wise
rows :: Tensor -> [Tensor]
rows = map T.squeezeAll . T.split 1 (T.Dim 0)

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

-- | Split Tensor into List of GPU Tensors
splitBatches :: Int -> Tensor -> [Tensor]
splitBatches bs = filter ((bs==) . head . T.shape) . T.split bs (Dim 0)
                . T.toDevice gpu

-- | Read Delimited data into Tensor
readDSV :: String -> FilePath -> IO ([String], Tensor)
readDSV sep path = do
    file <- lines <$> readFile path
    let col = drop 2 . splitOn sep $ head file
        dat = T.asTensor . map (map (read @Float) . drop 2 . splitOn sep) $ tail file
    pure (col, dat)

readCSV :: FilePath -> IO ([String], Tensor)
readCSV = readDSV ","

-- | Load all Transistor CSVs from a file path and concatenate them
loadCSVs :: FilePath -> IO ([String], Tensor)
loadCSVs path = do
    dats <- mapM readCSV csvs
    let (columns,_) = head dats
        values      = T.cat (Dim 0) $ map snd dats
    pure (columns, values)
  where
    csvs = map (\t -> path ++ "/" ++ show t ++ ".csv")  [(minBound :: Transistor) .. ]

-- | Read data from TXT file
readTSV :: FilePath -> IO ([String], Tensor)
readTSV path = do
    file <- filter ld . lines <$> readFile path
    let col = drop 2 . words $ head file
        dat = T.asTensor . map (map (read @Float) . drop 2 . words) $ tail file
    pure (col, dat)
  where
    ld "" = False
    ld l  = not $ ("%" `isPrefixOf` l) || ("---" `isInfixOf` l)

-- | Update nth value in a list
replaceNth :: Int -> a -> [a] -> [a]
replaceNth _ _   []   = []
replaceNth i n (x:xs) | i == 0    = n:xs
                      | otherwise = x:replaceNth (i-1) n xs

-- | Current Timestamp as formatted string
currentTimeStamp :: String -> IO String
currentTimeStamp format = formatTime defaultTimeLocale format <$> getCurrentTime

-- | Current Timestamp with default formatting: "%Y%m%d-%H%M%S"
currentTimeStamp' :: IO String
currentTimeStamp' = currentTimeStamp "%Y%m%d-%H%M%S"

-- | Create a model archive directory with time stamp and return path
createModelDir :: String -> IO String
createModelDir base = do
    path <- (path' ++) <$> currentTimeStamp'
    createDirectoryIfMissing True path
    pure path
  where
    path' = base ++ "/"

-- | Number of Random Latin Hyper Cube samples
lhsMaxMin :: Int -> Int-> Int -> IO Tensor 
lhsMaxMin dims samples num = do
    hc <- replicateM num (lhsRandom dims samples)
    let minDist    = T.stack (T.Dim 0) $ map (flip T.pdist 2.0) hc
        hypercubes = T.stack (T.Dim 0) hc
        maxminIdx  = T.asValue . T.argmax (T.Dim 0) T.KeepDim
                   . fst . T.maxDim (Dim 1) RemoveDim $ minDist
    pure $ T.select 0 maxminIdx hypercubes

-- | Random Latin Hyper Cube sample
lhsRandom :: Int -> Int -> IO Tensor
lhsRandom dims points = do
    hc' <- T.mulScalar inc . T.transpose2D . T.asTensor
         <$> replicateM dims (shuffle [0 .. (points' - 1)])
    rc' <- T.mulScalar inc <$> T.randIO' (T.shape hc')
    pure $ hc' + rc'
  where
    points' = realToFrac points :: Float
    inc     = 1.0 / points' :: Float
