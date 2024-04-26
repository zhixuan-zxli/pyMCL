#define PY_SSIZE_T_CLEAN
#include <algorithm>
#include <array>
#include <Python.h>
#include <numpy/arrayobject.h>

template <class T, std::size_t Dim>
static bool compare(const std::array<T, Dim> &a, const std::array<T, Dim> &b) {
    for(std::size_t i = 0; i < Dim; ++i) {
        if (a[i] != b[i])
            return a[i] < b[i];
    }
    return false;
}

template <class T, std::size_t Dim>
static void binsearchkw_work(const std::array<T, Dim> *cand, npy_intp num_cand, const std::array<T, Dim> *db, npy_intp num_db, npy_int32 *res) {
    for(std::size_t i = 0; i < (unsigned)num_cand; ++i) {
        auto resit = std::lower_bound(db, db + num_db, cand[i], compare<T, Dim>);
        if (resit != db + num_db && compare(cand[i], *resit) == false)
            res[i] = resit - db;
        else
            res[i] = -1;
    }
}

static PyObject *binsearchkw(PyObject *self, PyObject *args) {
    PyArrayObject *candidate, *database;

    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &database, &PyArray_Type, &candidate)) {
        PyErr_SetString(PyExc_TypeError, "Invalid argument: expecting 2 NumPy arrays.");
        return NULL;
    }

    if (PyArray_NDIM(candidate) != 2 || PyArray_NDIM(database) != 2) {
        PyErr_SetString(PyExc_ValueError, "Input array must be 2-dimensional.");
        return NULL;
    }

    int cand_type = PyArray_TYPE(candidate);
    npy_intp *cand_shape = PyArray_DIMS(candidate);
    void *cand_data = PyArray_DATA(candidate);

    int database_type = PyArray_TYPE(database);
    npy_intp *database_shape = PyArray_DIMS(database);
    void *database_data = PyArray_DATA(database);

    if (cand_type != database_type) {
        PyErr_SetString(PyExc_ValueError, "Data types mismatched. ");
        return NULL;
    }

    if (cand_shape[1] != database_shape[1] || cand_shape[1] < 1 || cand_shape[1] > 4) {
        PyErr_SetString(PyExc_ValueError, "The number of columns shoulbe be between 1 and 4. ");
        return NULL;
    }

    PyObject *result = PyArray_EMPTY(1, &cand_shape[0], NPY_INT32, 0/*fortran*/);
    if (!result) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for the result array.");
        return NULL;
    }

    void *result_data = PyArray_DATA((PyArrayObject *)result);

    if (cand_type == NPY_INT32) {
        if (cand_shape[1] == 1) {
            binsearchkw_work((std::array<npy_int32,1>*)cand_data, cand_shape[0], (std::array<npy_int32,1>*)database_data, database_shape[0], (npy_int32*)result_data);
        } else if(cand_shape[1] == 2) {
            binsearchkw_work((std::array<npy_int32,2>*)cand_data, cand_shape[0], (std::array<npy_int32,2>*)database_data, database_shape[0], (npy_int32*)result_data);
        } else if(cand_shape[1] == 3) {
            binsearchkw_work((std::array<npy_int32,3>*)cand_data, cand_shape[0], (std::array<npy_int32,3>*)database_data, database_shape[0], (npy_int32*)result_data);
        } else if(cand_shape[1] == 4) {
            binsearchkw_work((std::array<npy_int32,4>*)cand_data, cand_shape[0], (std::array<npy_int32,4>*)database_data, database_shape[0], (npy_int32*)result_data);
        }
    } else if(cand_type == NPY_INT64) {
        if (cand_shape[1] == 1) {
            binsearchkw_work((std::array<npy_int64,1>*)cand_data, cand_shape[0], (std::array<npy_int64,1>*)database_data, database_shape[0], (npy_int32*)result_data);
        } else if(cand_shape[1] == 2) {
            binsearchkw_work((std::array<npy_int64,2>*)cand_data, cand_shape[0], (std::array<npy_int64,2>*)database_data, database_shape[0], (npy_int32*)result_data);
        } else if(cand_shape[1] == 3) {
            binsearchkw_work((std::array<npy_int64,3>*)cand_data, cand_shape[0], (std::array<npy_int64,3>*)database_data, database_shape[0], (npy_int32*)result_data);
        } else if(cand_shape[1] == 4) {
            binsearchkw_work((std::array<npy_int64,4>*)cand_data, cand_shape[0], (std::array<npy_int64,4>*)database_data, database_shape[0], (npy_int32*)result_data);
        }
    } else if(cand_type == NPY_FLOAT64) {
        if (cand_shape[1] == 1) {
            binsearchkw_work((std::array<npy_double,1>*)cand_data, cand_shape[0], (std::array<npy_double,1>*)database_data, database_shape[0], (npy_int32*)result_data);
        } else if(cand_shape[1] == 2) {
            binsearchkw_work((std::array<npy_double,2>*)cand_data, cand_shape[0], (std::array<npy_double,2>*)database_data, database_shape[0], (npy_int32*)result_data);
        } else if(cand_shape[1] == 3) {
            binsearchkw_work((std::array<npy_double,3>*)cand_data, cand_shape[0], (std::array<npy_double,3>*)database_data, database_shape[0], (npy_int32*)result_data);
        } else if(cand_shape[1] == 4) {
            binsearchkw_work((std::array<npy_double,4>*)cand_data, cand_shape[0], (std::array<npy_double,4>*)database_data, database_shape[0], (npy_int32*)result_data);
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Unsupported data types. ");
        return NULL;
    }

    return result;
}

static PyMethodDef methods[] = {
    {"binsearchkw", binsearchkw, METH_VARARGS, "Binary search on multiple keywords"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "binsearchkw",
    NULL,
    -1,
    methods
};

extern "C" PyMODINIT_FUNC PyInit_binsearchkw(void) {
    import_array();
    return PyModule_Create(&moduledef);
}
