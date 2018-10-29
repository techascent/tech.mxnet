(ns tech.mxnet
  (:require [tech.datatype.base :as dtype-base]
            [tech.datatype.jna :as dtype-jna]
            [tech.resource :as resource]
            [tech.datatype.java-primitive :as primitive]
            [clojure.core.matrix.protocols :as mp]
            [tech.compute.driver :as drv]
            [tech.compute.registry :as registry]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.base :as base]
            [clojure.set :as c-set]
            [tech.compute.tensor.defaults :as ct-defaults])
  (:import [org.apache.mxnet NDArray DType]
           [com.sun.jna NativeLibrary Pointer Native Function]
           [com.sun.jna.ptr PointerByReference]
           [java.lang.reflect Field]
           [scala Enumeration$Value]))


(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)


(def ^:dynamic *native-library-name* "mxnet-scala-linux-x86_64-cpu")

(def load-native-library
  (memoize
   (fn [libname]
     (NativeLibrary/getInstance libname))))


(defn native-library
  [& [libname]]
  (load-native-library (or libname *native-library-name*)))


(def do-find-function
  (memoize
   (fn [^String fn-name ^String libname]
     (.getFunction ^NativeLibrary (native-library libname) fn-name))))


(defn find-function
  [fn-name & [libname]]
  (do-find-function fn-name (or libname *native-library-name*)))


(def array-fields
  (->> (.getDeclaredFields NDArray)
       (map (fn [^Field field]
              (.setAccessible field true)
              [(.getName field) field]))
       (into {})))

(defn handle-field
  ^Field[]
  (get array-fields "handle"))


(defn handle
  [^NDArray ary]
  (.getLong (handle-field) ary))


(def mxnet-datatype->datatype-map
  {(DType/Float32) :float32
   (DType/Float64) :float64
   (DType/Float16) :float16
   (DType/UInt8) :uint8
   (DType/Int32) :int32})


(defn mxnet-datatype->datatype
  [mxnet-dtype]
  (if-let [retval (get mxnet-datatype->datatype-map mxnet-dtype)]
    retval
    (throw (ex-info "Failed to map from integer to datatype"
                    {:value (.id ^Enumeration$Value mxnet-dtype)
                     :possible-values (->> (keys mxnet-datatype->datatype-map)
                                           (map #(.id ^Enumeration$Value %)))}))))


(defn get-array-datatype
  [^NDArray ary]
  (-> (.dtype ary)
      mxnet-datatype->datatype))


(defn allocate-native-ptr-storage
  ^Pointer []
  (->
   (case Native/POINTER_SIZE
     8 (dtype-jna/make-typed-pointer :int64 1)
     4 (dtype-jna/make-typed-pointer :int32 1))
   dtype-jna/->ptr-backing-store))


(defn number->ptr-size
  [num]
  (case Native/POINTER_SIZE
    8 (long num)
    4 (int num)))

(defn get-array-data
  ^PointerByReference [^NDArray ary]
  (let [handle (handle ary)
        data-ptr (PointerByReference.)
        data-fn (find-function "MXNDArrayGetData")
        retval (.invoke ^Function data-fn Integer (object-array
                                                   [(number->ptr-size handle)
                                                    data-ptr]))]
    (Pointer/nativeValue (.getValue data-ptr))))


(extend-type NDArray
  dtype-base/PDatatype
  (get-datatype [item] (get-array-datatype item))

  dtype-base/PAccess
  (set-value! [ptr ^long offset value]
    (dtype-base/set-value!
     (dtype-jna/->typed-pointer ptr)
     offset value))
  (set-constant! [ptr offset value elem-count]
    (dtype-base/set-constant! (dtype-jna/->typed-pointer ptr) offset value elem-count))
  (get-value [ptr ^long offset]
    (dtype-base/get-value (dtype-jna/->typed-pointer ptr) offset))

  mp/PDimensionInfo
  (dimensionality [m] (count (mp/get-shape m)))
  (get-shape [m] (ndarray/shape-vec m))
  (is-scalar? [m] false)
  (is-vector? [m] true)
  (dimension-count [m dimension-number]
    (let [shape (mp/get-shape m)]
      (if (<= (count shape) (long dimension-number))
        (get shape dimension-number)
        (throw (ex-info "Array does not have specific dimension"
                        {:dimension-number dimension-number
                         :shape shape})))))

  resource/PResource
  (release-resource [ary] (.dispose ary))

  mp/PElementCount
  (element-count [ary] (apply * (ndarray/shape-vec ary)))

  dtype-base/PContainerType
  ;;Do jna buffer to take advantage of faster memcpy, memset, and
  ;;other things jna datatype bindings provide.
  (container-type [ptr] :jna-buffer)

  dtype-base/PCopyRawData
  (copy-raw->item! [raw-data ary-target target-offset options]
    (dtype-base/copy-raw->item! (dtype-jna/->typed-pointer raw-data) ary-target
                                target-offset options))

  dtype-jna/PToPtr
  (->ptr-backing-store [item]
    (dtype-jna/make-jna-pointer (get-array-data item)))

  primitive/PToBuffer
  (->buffer-backing-store [src]
    (primitive/->buffer-backing-store (dtype-jna/->typed-pointer src)))

  primitive/PToArray
  (->array [src] nil)
  (->array-copy [src]
    (primitive/->array-copy (dtype-jna/->typed-pointer src)))

  drv/PDeviceProvider
  (get-device [ary]
    (drv/get-device (dtype-jna/->typed-pointer ary)))

  drv/PDriverProvider
  (get-driver [ary]
    (drv/get-driver (dtype-jna/->typed-pointer ary))))


(defn datatype->mxnet-datatype
  [dtype]
  (let [rev-map (c-set/map-invert mxnet-datatype->datatype-map)]
    (if-let [retval (get rev-map dtype)]
      retval
      (throw (ex-info "No know datatype mapping"
                      {:datatype dtype
                       :mxnet-datatypes (keys rev-map)})))))


(defn empty-array
  [shape & {:keys [datatype ctx]}]
  (let [mxnet-dtype (datatype->mxnet-datatype
                     (ct-defaults/datatype datatype))]
    (resource/track
     (ndarray/empty shape (merge {:dtype mxnet-dtype}
                                 (when ctx
                                   {:ctx ctx}))))))
