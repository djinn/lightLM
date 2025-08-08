#include <Python.h>
#include <structmember.h>
#include "lightlm/args.h"
#include "lightlm/vector.h"
#include "lightlm/densematrix.h"
#include "lightlm/dictionary.h"

// ==========================================================================
// Args Type
// ==========================================================================

typedef struct {
    PyObject_HEAD
    lightlm_args_t args;
} LightlmArgsObject;

static int
LightlmArgs_init(LightlmArgsObject *self, PyObject *args, PyObject *kwds)
{
    lightlm_args_t* temp_args = lightlm_args_new();
    if (temp_args == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate temporary lightlm_args_t");
        return -1;
    }
    self->args = *temp_args;
    free(temp_args);
    return 0;
}

static void
LightlmArgs_dealloc(LightlmArgsObject *self)
{
    free(self->args.input);
    free(self->args.output);
    free(self->args.label);
    free(self->args.pretrainedVectors);
    free(self->args.autotuneValidationFile);
    free(self->args.autotuneMetric);
    free(self->args.autotuneModelSize);
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyObject*
LightlmArgs_get_minCount(LightlmArgsObject *self, void *closure)
{
    return PyLong_FromLong(self->args.minCount);
}

static int
LightlmArgs_set_minCount(LightlmArgsObject *self, PyObject *value, void *closure)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the minCount attribute");
        return -1;
    }
    if (!PyLong_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "The minCount attribute value must be an integer");
        return -1;
    }
    self->args.minCount = PyLong_AsLong(value);
    return 0;
}

static PyObject *
LightlmArgs_hello(LightlmArgsObject *self, PyObject *Py_UNUSED(ignored))
{
    printf("Hello from lightlm.Args object!\n");
    Py_RETURN_NONE;
}

static PyMethodDef LightlmArgs_methods[] = {
    {"hello", (PyCFunction) LightlmArgs_hello, METH_NOARGS, "Prints a hello message."},
    {NULL}  /* Sentinel */
};

static PyTypeObject LightlmArgsType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "lightlm.Args",
    .tp_doc = "lightlm Args object",
    .tp_basicsize = sizeof(LightlmArgsObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc) LightlmArgs_init,
    .tp_dealloc = (destructor) LightlmArgs_dealloc,
    .tp_methods = LightlmArgs_methods,
};


// ==========================================================================
// Vector Type
// ==========================================================================

typedef struct {
    PyObject_HEAD
    lightlm_vector_t *vec;
} LightlmVectorObject;

static int
LightlmVector_init(LightlmVectorObject *self, PyObject *args, PyObject *kwds)
{
    long long size;
    if (!PyArg_ParseTuple(args, "L", &size)) {
        return -1;
    }
    self->vec = lightlm_vector_new(size);
    if (self->vec == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate lightlm_vector_t");
        return -1;
    }
    return 0;
}

static void
LightlmVector_dealloc(LightlmVectorObject *self)
{
    lightlm_vector_free(self->vec);
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static Py_ssize_t
LightlmVector_len(LightlmVectorObject *self)
{
    return self->vec->size;
}

static PyObject *
LightlmVector_zero(LightlmVectorObject *self, PyObject *Py_UNUSED(ignored))
{
    lightlm_vector_zero(self->vec);
    Py_RETURN_NONE;
}

static PyMethodDef LightlmVector_methods[] = {
    {"zero", (PyCFunction) LightlmVector_zero, METH_NOARGS, "Set all elements of the vector to zero."},
    {NULL}  /* Sentinel */
};

static PySequenceMethods LightlmVector_as_sequence = {
    .sq_length = (lenfunc)LightlmVector_len,
};

static PyTypeObject LightlmVectorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "lightlm.Vector",
    .tp_doc = "lightlm Vector object",
    .tp_basicsize = sizeof(LightlmVectorObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc) LightlmVector_init,
    .tp_dealloc = (destructor) LightlmVector_dealloc,
    .tp_as_sequence = &LightlmVector_as_sequence,
    .tp_methods = LightlmVector_methods,
};


// ==========================================================================
// Matrix Type
// ==========================================================================

typedef struct {
    PyObject_HEAD
    lightlm_matrix_t *mat;
} LightlmMatrixObject;

static int
LightlmMatrix_init(LightlmMatrixObject *self, PyObject *args, PyObject *kwds)
{
    long long m, n;
    if (!PyArg_ParseTuple(args, "LL", &m, &n)) {
        return -1;
    }
    // For now, we only create dense matrices
    self->mat = lightlm_dense_matrix_new(m, n);
    if (self->mat == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate lightlm_matrix_t");
        return -1;
    }
    return 0;
}

static void
LightlmMatrix_dealloc(LightlmMatrixObject *self)
{
    if (self->mat) {
        self->mat->free(self->mat);
    }
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyObject *
LightlmMatrix_rows(LightlmMatrixObject *self, void *closure)
{
    return PyLong_FromLongLong(self->mat->m_);
}

static PyObject *
LightlmMatrix_cols(LightlmMatrixObject *self, void *closure)
{
    return PyLong_FromLongLong(self->mat->n_);
}

static PyObject *
LightlmMatrix_zero(LightlmMatrixObject *self, PyObject *Py_UNUSED(ignored))
{
    lightlm_dense_matrix_zero(self->mat);
    Py_RETURN_NONE;
}

static PyGetSetDef LightlmMatrix_getseters[] = {
    {"rows", (getter) LightlmMatrix_rows, NULL, "number of rows", NULL},
    {"cols", (getter) LightlmMatrix_cols, NULL, "number of columns", NULL},
    {NULL}  /* Sentinel */
};

static PyMethodDef LightlmMatrix_methods[] = {
    {"zero", (PyCFunction) LightlmMatrix_zero, METH_NOARGS, "Set all elements of the matrix to zero."},
    {NULL}  /* Sentinel */
};

static PyTypeObject LightlmMatrixType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "lightlm.Matrix",
    .tp_doc = "lightlm Matrix object",
    .tp_basicsize = sizeof(LightlmMatrixObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc) LightlmMatrix_init,
    .tp_dealloc = (destructor) LightlmMatrix_dealloc,
    .tp_methods = LightlmMatrix_methods,
    .tp_getset = LightlmMatrix_getseters,
};


// ==========================================================================
// Dictionary Type
// ==========================================================================

typedef struct {
    PyObject_HEAD
    lightlm_dictionary_t *dict;
} LightlmDictionaryObject;

static int
LightlmDictionary_init(LightlmDictionaryObject *self, PyObject *args, PyObject *kwds)
{
    LightlmArgsObject *args_obj;
    if (!PyArg_ParseTuple(args, "O!", &LightlmArgsType, &args_obj)) {
        return -1;
    }
    self->dict = lightlm_dictionary_new(&args_obj->args);
    if (self->dict == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate lightlm_dictionary_t");
        return -1;
    }
    return 0;
}

static void
LightlmDictionary_dealloc(LightlmDictionaryObject *self)
{
    lightlm_dictionary_free(self->dict);
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyObject *
LightlmDictionary_read_from_file(LightlmDictionaryObject *self, PyObject *args)
{
    const char *filename;
    if (!PyArg_ParseTuple(args, "s", &filename)) {
        return NULL;
    }
    lightlm_dictionary_read_from_file(self->dict, filename);
    Py_RETURN_NONE;
}

static PyObject *
LightlmDictionary_nwords(LightlmDictionaryObject *self, void *closure)
{
    return PyLong_FromLong(lightlm_dictionary_nwords(self->dict));
}

static PyObject *
LightlmDictionary_nlabels(LightlmDictionaryObject *self, void *closure)
{
    return PyLong_FromLong(lightlm_dictionary_nlabels(self->dict));
}

static PyObject *
LightlmDictionary_ntokens(LightlmDictionaryObject *self, void *closure)
{
    return PyLong_FromLongLong(lightlm_dictionary_ntokens(self->dict));
}

static PyGetSetDef LightlmDictionary_getseters[] = {
    {"nwords", (getter) LightlmDictionary_nwords, NULL, "number of words", NULL},
    {"nlabels", (getter) LightlmDictionary_nlabels, NULL, "number of labels", NULL},
    {"ntokens", (getter) LightlmDictionary_ntokens, NULL, "number of tokens", NULL},
    {NULL}  /* Sentinel */
};

static PyMethodDef LightlmDictionary_methods[] = {
    {"read_from_file", (PyCFunction) LightlmDictionary_read_from_file, METH_VARARGS, "Read a vocabulary from a file."},
    {NULL}  /* Sentinel */
};

static PyTypeObject LightlmDictionaryType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "lightlm.Dictionary",
    .tp_doc = "lightlm Dictionary object",
    .tp_basicsize = sizeof(LightlmDictionaryObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc) LightlmDictionary_init,
    .tp_dealloc = (destructor) LightlmDictionary_dealloc,
    .tp_methods = LightlmDictionary_methods,
    .tp_getset = LightlmDictionary_getseters,
};


// ==========================================================================
// Module Definition
// ==========================================================================

static PyModuleDef lightlm_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "lightlm",
    .m_doc = "Python interface for the lightlm C library",
    .m_size = -1,
};

PyMODINIT_FUNC
PyInit_lightlm(void)
{
    PyObject *m;

    // Prepare the Args type
    if (PyType_Ready(&LightlmArgsType) < 0)
        return NULL;

    // Prepare the Vector type
    if (PyType_Ready(&LightlmVectorType) < 0)
        return NULL;

    // Prepare the Matrix type
    if (PyType_Ready(&LightlmMatrixType) < 0)
        return NULL;

    // Prepare the Dictionary type
    if (PyType_Ready(&LightlmDictionaryType) < 0)
        return NULL;

    m = PyModule_Create(&lightlm_module);
    if (m == NULL)
        return NULL;

    // Add the Args type to the module
    Py_INCREF(&LightlmArgsType);
    if (PyModule_AddObject(m, "Args", (PyObject *) &LightlmArgsType) < 0) {
        Py_DECREF(&LightlmArgsType);
        Py_DECREF(m);
        return NULL;
    }

    // Add the Vector type to the module
    Py_INCREF(&LightlmVectorType);
    if (PyModule_AddObject(m, "Vector", (PyObject *) &LightlmVectorType) < 0) {
        Py_DECREF(&LightlmVectorType);
        Py_DECREF(&LightlmArgsType);
        Py_DECREF(m);
        return NULL;
    }

    // Add the Matrix type to the module
    Py_INCREF(&LightlmMatrixType);
    if (PyModule_AddObject(m, "Matrix", (PyObject *) &LightlmMatrixType) < 0) {
        Py_DECREF(&LightlmMatrixType);
        Py_DECREF(&LightlmVectorType);
        Py_DECREF(&LightlmArgsType);
        Py_DECREF(m);
        return NULL;
    }

    // Add the Dictionary type to the module
    Py_INCREF(&LightlmDictionaryType);
    if (PyModule_AddObject(m, "Dictionary", (PyObject *) &LightlmDictionaryType) < 0) {
        Py_DECREF(&LightlmDictionaryType);
        Py_DECREF(&LightlmMatrixType);
        Py_DECREF(&LightlmVectorType);
        Py_DECREF(&LightlmArgsType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
