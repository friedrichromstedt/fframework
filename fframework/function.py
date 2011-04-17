"""
*   Defining operator overloads requires that the operator Functions are at
    hand.  But those are derivates of ``Function``.  So we would have to
    define the Functions for operator overloads in the same scope as the
    Function base class.  But this we want to avoid, because the bare Function
    class module should stay bare.

*   Defining the ``Function`` class alone, it is possible to define 
    mathematical Functions in ``op``, when defining the ``OpFunction`` class, 
    which has operator overloads using those operator Functions.

*   Tests whether an object is a function or not require only testing if the
    object is an instance of ``Function``.
"""

__all__ = ['Function', 'asfunction']

class Function:
    """The base class of all Functions.  It is a bare class without any 
    attributes.  Used in ``isinstance(object, Function)``."""

    def __init__(self):
        pass

class Constant(Function):
    """
    A Function yielding always the same value.
    """

    def __init__(self, value):
        """
        *value* is the value of the Constant.
        """

        self.value = value

    def __call__(self, *args, **kwargs):
        """Returns the constant value."""
        return self.value

class Identity(Function):
    """
    Returns always its argument(s).  If called with precisely one 
    argument, returns the argument as a scalar, else the argument vector
    is returned.
    """

    def __call__(self, *args):
        if len(args) == 1:
            return args[0]
        else:
            return args

def asfunction(function_like):
    """
    *   If *function_like* is a :class:`Function`, it is returned unchanged.
    *   Else, the *function_like* is interpreted as a :class:`Constant`.
    """

    if isinstance(function_like, Function):
        return function_like
    else:
        return Constant(function_like)
