#define PY_SSIZE_T_CLEAN
#include <algorithm>
#include <array>
#include <Python.h>
#include <numpy/arrayobject.h>

template <std::size_t Dim>
static bool compare(const std::array<npy_int, Dim> &a, const std::array<npy_int, Dim> &b) {
    for(std::size_t i = 0; i < Dim; ++i) {
        if (a[i] != b[i])
            return a[i] < b[i];
    }
    return false;
}

template <std::size_t Dim>
static void binsearchkw_work(const std::array<npy_int, Dim> *cand, npy_intp num_cand, const std::array<npy_int, Dim> *db, npy_intp num_db, npy_int *res) {
    for(std::size_t i = 0; i < (unsigned)num_cand; ++i) {
        auto resit = std::lower_bound(db, db + num_db, cand[i], compare<Dim>);
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
    
    if (PyArray_TYPE(candidate) != NPY_INT || PyArray_TYPE(database) != NPY_INT) {
        PyErr_SetString(PyExc_ValueError, "Expect int array. ");
        return NULL;
    }

    npy_intp *cand_shape = PyArray_DIMS(candidate);
    void *cand_data = PyArray_DATA(candidate);

    npy_intp *database_shape = PyArray_DIMS(database);
    void *database_data = PyArray_DATA(database);

    if (cand_shape[1] < 1 || cand_shape[1] > 4 || database_shape[1] != cand_shape[1]) {
        PyErr_SetString(PyExc_ValueError, "Incorrect entity dimension. ");
        return NULL;
    }

    PyObject *result = PyArray_EMPTY(1, &cand_shape[0], NPY_INT, 0);
    if (!result) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for the result array.");
        return NULL;
    }

    void *result_data = PyArray_DATA((PyArrayObject *)result);

    if (cand_shape[1] == 1) {
        binsearchkw_work<1>((std::array<npy_int,1>*)cand_data, cand_shape[0], (std::array<npy_int,1>*)database_data, database_shape[0], (npy_int*)result_data);
    } else if(cand_shape[1] == 2) {
        binsearchkw_work<2>((std::array<npy_int,2>*)cand_data, cand_shape[0], (std::array<npy_int,2>*)database_data, database_shape[0], (npy_int*)result_data);
    } else if(cand_shape[1] == 3) {
        binsearchkw_work<3>((std::array<npy_int,3>*)cand_data, cand_shape[0], (std::array<npy_int,3>*)database_data, database_shape[0], (npy_int*)result_data);
    } else if(cand_shape[1] == 4) {
        binsearchkw_work<4>((std::array<npy_int,4>*)cand_data, cand_shape[0], (std::array<npy_int,4>*)database_data, database_shape[0], (npy_int*)result_data);
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
