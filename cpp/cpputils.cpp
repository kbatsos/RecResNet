#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <string.h>
#include <iostream>
#include <stdint.h>
#include <png++/png.hpp>
#include <boost/python.hpp>
#include "boost/python/extract.hpp"
#include "boost/python/numeric.hpp"
#include <numpy/ndarrayobject.h>

using namespace std;
using namespace boost::python;

typedef uint16_t uint16;


PyObject* DDSupport(PyObject* dm){
	PyArrayObject* dmA = reinterpret_cast<PyArrayObject*>(dm);

	float * dmp = reinterpret_cast<float*>(PyArray_DATA(dmA));

	npy_intp *shape = PyArray_DIMS(dmA);


	PyObject* res = PyArray_SimpleNew(2,PyArray_DIMS(dmA), NPY_FLOAT);
	float* res_data = static_cast<float*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(res)));

	bool* ddmap = (bool *)calloc((int)shape[0]*shape[1],sizeof(bool));

    for(int j=0; j<shape[1]; j++){
            ddmap[j] = true;
            ddmap[(shape[0]-1)*(shape[1]-1)+j] = true;
    }

    for(int i=0;i<shape[0];i++){
            ddmap[i*shape[1]] = true;
            ddmap[i*shape[1]+(shape[0]-1)] = true;
    }

	#pragma omp parallel
	{

			#pragma omp for
			for(int i=0; i<shape[0]; i++){
				ddmap[i*shape[1]] = true;
				ddmap[i*shape[1]+(shape[1]-1)] = true;
			}

	}


	#pragma omp parallel
	{

			#pragma omp for
			for(int j=0; j<shape[1]; j++){
				ddmap[j] = true;
				ddmap[(shape[0]-1)*shape[1]+j] = true;
			}

	}



	#pragma omp parallel
	{

			#pragma omp for
			for(int i=1; i<shape[0]-1; i++){
					for(int j=1; j<shape[1]-1; j++){
							float val = dmp[i*shape[1]+j];

							if(fabs( dmp[(i-1)*shape[1]+j] - val ) > 1 ||
							   fabs( dmp[(i+1)*shape[1]+j] - val ) > 1 ||
							   fabs( dmp[i*shape[1]+(j-1)] - val ) > 1 ||
							   fabs( dmp[i*shape[1]+(j+1)] - val ) > 1){
								ddmap[i*shape[1]+j] = true;

							}

					}
			}

	}


	#pragma omp parallel
	{

			#pragma omp for
			for(int i=0; i<shape[0]; i++){
					for(int j=0; j<shape[1]; j++){

							if(ddmap[i*shape[1]+j]){
								res_data[ i*shape[1]+j ] =0;
								continue;
							}


							int leftdist=0;
							for(int k=j; k>=0;k--){
									if(ddmap[i*shape[1]+k]){
											leftdist=j-k;
											break;
									}

							}
							int rightdist=0;
							for(int k=j; k<shape[1];k++){
									if(ddmap[i*shape[1]+k]){
											rightdist=k-j;
											break;
									}
							}

							int topdist=0;
							for(int k=i; k>=0;k--){
									if(ddmap[k*shape[1]+j]){
											topdist=i-k;
											break;
									}
							}

							int bottomdist=0;
							for(int k=i; k<shape[1];k++){
									if(ddmap[k*shape[1]+j]){
											bottomdist =k-i;
											break;
									}
							}


							int max = leftdist;

							if(rightdist > max)
								max = rightdist;
							if(topdist > max)
								max = topdist;
							if(bottomdist > max)
								max = bottomdist;

							res_data[ i*shape[1]+j] = max;


					}
			}

	}

delete [] ddmap;

return res;

}





template <typename T>
inline T getDisp (T* data_,const int width,const int32_t u,const int32_t v) {
  return data_[v*width+u];
}

template <typename T>
// is disparity valid
inline bool isValid (T* data_,const int32_t u,const int32_t v,const int width) {
  return data_[v*width+u]>=0;
}

void write2png(PyObject *disp, const std::string & path  ){

	PyArrayObject* dispA = reinterpret_cast<PyArrayObject*>(disp);
	float * dispD = reinterpret_cast<float*>(PyArray_DATA(dispA));
	npy_intp *shape = PyArray_DIMS(dispA);

	int height = shape[0]; int width = shape[1];

    png::image< png::gray_pixel_16 > image(width,height);
    for (int32_t v=0; v<height; v++) {
      for (int32_t u=0; u<width; u++) {
        if (isValid(dispD,u,v,width)) image.set_pixel(u,v,(uint16_t)(std::max(getDisp(dispD,width,u,v)*256.0,1.0)));
        else              image.set_pixel(u,v,0);
      }
    }
    image.write(path);


}

PyObject* make_occ(PyObject *l_gt, PyObject *r_gt){


    PyArrayObject* l_gtA = reinterpret_cast<PyArrayObject*>(l_gt);
    PyArrayObject* r_gtA = reinterpret_cast<PyArrayObject*>(r_gt);

    //Get the pointer to the data
    float * lgtD = reinterpret_cast<float*>(PyArray_DATA(l_gtA));
    float * rgtD = reinterpret_cast<float*>(PyArray_DATA(r_gtA));


    npy_intp *shape = PyArray_DIMS(l_gtA);
    npy_intp *shapeout = new npy_intp[2];
    shapeout[0] = shape[0]; shapeout[1] = shape[1];

	 PyObject* res = PyArray_SimpleNew(2, shapeout, NPY_FLOAT);

	 //Get the pointer to the data
	 float* res_data = static_cast<float*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(res)));

	#pragma omp parallel for
	 for(int i=0; i<shape[0]; i++){
		 for(int j=0; j<shape[1]; j++){
			 float d = lgtD[ i*shape[1] +j ];
			 int dr = (int)round(d);
			 if( round( j- lgtD[ i*shape[1] +j ]  ) <0  )
				 res_data[ i*shape[1] +j ] =0;
			 else if( abs ( (int) round( rgtD[ i*shape[1] +  (j- dr ) ] ) - dr ) >1   ){
				 res_data[ i*shape[1] +j ] =0;
			 }else
				 res_data[ i*shape[1] +j ] =lgtD[ i*shape[1] +j ] ;
		 }
	 }


	 return res;


}




BOOST_PYTHON_MODULE(cpputils) {

    numeric::array::set_module_and_type("numpy", "ndarray");

    def("write2png",write2png);
    def("make_occ",make_occ);


    import_array();
}

