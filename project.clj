(defproject techascent/tech.mxnet "0.1.0-SNAPSHOT"
  :description "mxnet bindings to the tech.compute system."
  :url "http://github.com/tech-ascent/tech.mxnet"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.9.0"]
                 [org.apache.mxnet.contrib.clojure/clojure-mxnet-linux-cpu "1.3.0"]
                 [techascent/tech.compute "1.10"]]

  :profiles {:dev
             ;;Unit tests need this.
             {:dependencies [[techascent/tech.opencv "1.2"]]}})
