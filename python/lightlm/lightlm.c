#include <Python.h>
#include "lightlm/args.h"

typedef struct {
    PyObject_HEAD
    lightlm_args_t *args;
} LightlmArgsObject;

static int
LightlmArgs_init(LightlmArgsObject *self, PyObject *args, PyObject *kwds)
{
    self->args = lightlm_args_new();
    if (self->args == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate lightlm_args_t");
        return -1;
    }
    return 0;
}

static void
LightlmArgs_dealloc(LightlmArgsObject *self)
{
    lightlm_args_free(self->args);
    Py_TYPE(self)->tp_free((PyObject *) self);
}

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
};

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
    if (PyType_Ready(&LightlmArgsType) < 0)
        return NULL;

    m = PyModule_Create(&lightlm_module);
    if (m == NULL)
        return NULL;

    Py_INCREF(&LightlmArgsType);
    if (PyModule_AddObject(m, "Args", (PyObject *) &LightlmArgsType) < 0) {
        Py_DECREF(&LightlmArgsType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
