(ns tech.mxnet-test
  (:require [clojure.test :refer :all]
            [tech.mxnet :as mxnet]
            [tech.opencv :as opencv]
            [tech.datatype :as dtype]))


(deftest set-get-values
  (let [test-img (opencv/load "test/data/test.jpg")
        src-data (mxnet/empty-array (dtype/shape test-img) :datatype :float32)
        short-data (short-array (dtype/ecount test-img))]
    (dtype/copy! test-img src-data)
    (dtype/copy! src-data short-data)
    (is (=  (list 172 170 170 172 170 170 171 169 169 171)
            (take 10 (dtype/->vector short-data))))))
