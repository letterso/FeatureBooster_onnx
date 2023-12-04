#include <pybind11/pybind11.h>

#include <sstream> 

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "opencv_type_casters.h"
#include "ORBextractor.h"

namespace py = pybind11;
using namespace pybind11::literals;

using namespace ORB_SLAM;

PYBIND11_MODULE(orbslam3_features, m) 
{
    // optional module docstring
    m.doc() = "pybind11 plugin for ORBSLAM3 features";

    //declareCvTypes(m); // just experimental 
    
    // bindings to ORBextractor class
    py::class_<ORBextractor>(m, "ORBextractor")
        .def(py::init<int, float, int, int, int>(),"nfeatures"_a, "scaleFactor"_a, "nlevels"_a, "iniThFAST"_a=20, "minThFAST"_a=6)
        .def("GetNumFeatures", &ORBextractor::GetNumFeatures)  
        .def("GetLevels", &ORBextractor::GetLevels)
        .def("GetScaleFactor", &ORBextractor::GetScaleFactor)  
        .def("detectAndCompute",   
            [](ORBextractor& o, cv::Mat& image) 
            { 
                cv::Mat mask = cv::Mat();  // input mask is not actually used by the implementation 
                std::vector<cv::KeyPoint> keypoints; 
                cv::Mat descriptors;     
                std::vector<int> vLapping = {0, 1000};          
                int features_num = o.detectAndCompute(image, mask, keypoints, descriptors, vLapping); 
                return std::make_tuple(keypoints, descriptors);
            })                                
        .def("__repr__",
            [](const ORBextractor &o) 
            {
                std::stringstream ss;
                ss << "<orb_features.ORBextractor - #features: " << o.GetNumFeatures() << ", #levels: " << o.GetLevels() << ", factor: " << o.GetScaleFactor() << ">";
                return ss.str();
            });  

}
