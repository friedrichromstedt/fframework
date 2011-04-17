from fframework.function import Function, Constant, Identity, \
    asfunction
try:
    import numpy
    numpy_available = True
except ImportError:
    # Use the Python math module as fallback.
    import math
    numpy_available = False

# All other Function can be accessed more clearly by ordinary means or by
# using numpy functions (numpy.sin, numpy.cos):
__all__ = ['OpFunction', 'asopfunction', 'compound', 'InBetween', 'Not', 
    'Cos', 'Sin', 'Exp', 'Sqrt', 'SumCall', 'Indexing', 'Attribute', 'AsType',
    'Clip', 'Int', 'Float', 'Bool']

class OpFunction(Function):
    """
    Supports overloaded primitve arithmetics.  Examples:

    1)  ``fa + fb``, etc.
    2)  ``numpy.cos(fangle)``.  This works through the :meth:`cos` method.
        Same for ``numpy.sin(fangle)``.
    3)  ``numpy.sum(farray, axis=1)``.  This works through the :meth:`sum`
        method.
    """

    def __add__(self, other):
        """Returns the :class:`Sum` with another Function."""

        other = asfunction(other)
        return Sum(self, other)

    def __radd__(self, other):
        """Returns the :class:`Sum` with another Function."""

        other = asfunction(other)
        return Sum(other, self)

    def __sub__(self, other):
        """Returns the :class:`Sum` with the negative of the other 
        Function."""

        other = asfunction(other)
        return Sum(self, Neg(other))

    def __rsub__(self, other):
        """Returns the :class:`Sum` of the other Function with the negative of
        ``self``."""

        other = asfunction(other)
        return Sum(other, -self)

    def __mul__(self, other):
        """Returns the :class:`Product` with the other Function."""

        other = asfunction(other)
        return Product(self, other)

    def __rmul__(self, other):
        """Returns the :class:`Product` of the other Function with 
        ``self``."""

        other = asfunction(other)
        return Product(other, self)

    def __div__(self, other):
        """Returns the :class:`Quotient` with the other Function."""

        other = asfunction(other)
        return Quotient(self, other)

    def __rdiv__(self, other):
        """Returns the :class:`Quotient` of the other Function with 
        ``self``."""

        other = asfunction(other)
        return Quotient(other, self)

    def __pow__(self, exponent):
        """Raises ``self`` to the power of the other Function."""

        exponent = asfunction(exponent)
        return Power(base=self, exponent=exponent)

    def __rpow__(self, base):
        """Raises the other Function to the power of ``self``."""

        base = asfunction(base)
        return Power(base=base, exponent=self)

    def __pos__(self):
        """Returns ``self``."""

        return self

    def __neg__(self):
        """Returns the negative of ``self``."""

        return self | Neg()

    def __int__(self):
        """Returns an integer-function."""

        return self | Int()

    def __float__(self):
        """Returns a float-function."""

        return self | Float()

    def __bool__(self):
        """Returns a bool-function."""

        return self | Bool()

    def sin(self):
        """Takes the sine."""

        return self | Sin()

    def cos(self):
        """Takes the cosine."""

        return self | Cos()

    def exp(self):
        """Exponentiates."""

        return self | Exp()

    def sqrt(self):
        """Takes the square root."""

        return self | Sqrt()

    def sum(self, *args, **kwargs):
        """Returns::
            
            self | SumCall(*args, **kwargs)
        
        This will hand *args* and *kwargs* over to self's value's ``.sum()``
        method."""

        return self | SumCall(*args, **kwargs)

    def clip(self, low, high):
        """Returns a :class:`Clip` instance of ``self`` with *low* and *high* 
        set up."""

        return Clip(leaf=self, low=low, high=high)

    def astype(self, *args, **kwargs):
        """Returns the following::

            self | AsType(*args, **kwargs)

        This can be called and will convert the result of ``self`` on-the-fly
        via its ``.dtype()``.  *args* and *kwargs* will be handed over to
        ``.dtype()``, this is done by the :class:`AsType` instance."""

        return self | AsType(*args, **kwargs)

    def __getitem__(self, index):
        """Returns::
        
            self | Indexing(index)
        """

        return self | Indexing(index)

    def __or__(self, other):
        """Piping is composition of Functions.  The pipe operator is designed
        such that the wrapping function is written last: *other* will be 
        executed with the ouput of *self* as input."""

        return ComposedFunction(a=self, b=other)

    #
    # We do not define the other binary-operators, because they interfer with
    # the use of | (binary OR) as the piping syntax.
    #

    def __lt__(self, other):
        """*self* < *other*"""

        return Less(self, other)

    def __le__(self, other):
        """*self* <= *other*"""

        return LessEqual(self, other)

    def __eq__(self, other):
        """*self* == *other*"""

        return Equal(self, other)

    def __ne__(self, other):
        """*self* != *other*"""

        return NotEqual(self, other)

    def __gt__(self, other):
        """*self* > *other*"""

        return Greater(self, other)

    def __ge__(self, other):
        """*self* >= *other*"""

        return GreaterEqual(self, other)

class OpWrap(OpFunction):
    """
    Calls another Function, but provides additionally the full operator
    overloading of :class:`OpFunction`.
    """

    def __init__(self, target):
        """*target* is an ordinary Function, being used during __call__."""

        self.target = target

    def __call__(self, *args, **kwargs):
        """Calls just ``.wrap``."""

        return self.target(*args, **kwargs)

class Apply(OpFunction):
    """
    Abstract class where apply-like function might derive from.  Apply-like
    functions call some function on the argument or call some method of
    the argument, and are usually used in piping syntax::

        applicant | Applier(...)
    """
    
    def __init__(self, *args, **kwargs):
        """*args* and *kwargs* are not passed through the Function converting
        machine.  Instead, they are just stored inside.  They might be used
        inside of ``__call__()``."""

        self.args = args
        self.kwargs = kwargs

class ComposedFunction(OpFunction):
    """Executes one function with the output of another."""

    def __init__(self, a, b):
        """*b* will be executed with the output of *a* as input."""

        self.a = asfunction(a)
        self.b = asfunction(b)
    
    def __call__(self, *args, **kwargs):
        """Returns ``b(a(...))``."""

        return self.b(self.a(*args, **kwargs))

class OpConstant(Constant, OpFunction):
    """:class:`~fframework.function.Constant`, extended by mathematical
    overloads."""

    pass

class OpIdentity(Identity, OpFunction):
    """:class:`~fframework.function.Identity`, extended by mathematical
    overloads."""

    pass

def asopfunction(opfunction_like):
    """
    *   If *opfunction_like* is a :class:`OpFunction`, it is returned 
        unchanged.
    *   If *opfunction_like* is a :class:`~fframework.function.Function`,
        ``OpWrap(opfunction_like)`` is returned.
    *   Else, the *opfunction_like* is interpreted as a :class:`OpConstant`
        instance.
    """

    if isinstance(opfunction_like, OpFunction):
        return opfunction_like
    elif isinstance(opfunction_like, Function):
        return OpWrap(opfunction_like)
    else:
        return OpConstant(opfunction_like)

class _List(OpFunction):
    """
    Constructs a list from a number of list items.
    """

    def __init__(self, elements):
        """*elements* are the elements of the resulting list.  All elements
        of *elements* are passed through :func:`asfunction`."""

        self._elements = map(asfunction, elements)

    def __call__(self, *args, **kwargs):
        """Calls all elements, and constructs a list from the call results."""

        return [element(*args, **kwargs) for element in self._elements]

class _Tuple(OpFunction):
    """
    Constructs a tuple from a number of tuple items.
    """

    def __init__(self, elements):
        """*elements* are the elements of the resulting tuple.  All elements
        of *elements* are passed through :func:`asfunction`."""

        self._elements = map(asfunction, elements)

    def __call__(self, *args, **kwargs):
        """Calls all elements, and constructs a tuple from the call 
        results."""

        return tuple([element(*args, **kwargs) for element in self._elements])

class _Dict(OpFunction):
    """
    Constructs a dictionary from keys and values.
    """

    def __init__(self, dictionary):
        """*dictionary* is a dict whose keys and values will be converted
        by :func:`asfunction`."""

        self._keys = map(asfunction, dictionary.keys())
        self._values = map(asfunction, dictionary.values())

    def __call__(self, *args, **kwargs):
        """Calls all keys and values, and constructs a dict from the call
        results."""

        keys = [key(*args, **kwargs) for key in self._keys]
        values = [value(*args, **kwargs) for value in self._values]
        return dict(zip(keys, values))

def compound(obj):
    """Replaces lists, tuples, dicts, constants in *obj* by corresponding 
    Functions.

    *   lists are replaced by :class:`_List` instances.
    *   tuples are replaced by :class:`_Tuple` instances.
    *   dicts are replaced by :class:`_Dict` instances.
    *   Other objects are passed through :func:`asfunction`.
    
    Use this method to generate Functions out of lists, tuples, or
    dictionary composed of other Functions."""

    if isinstance(obj, list):
        return _List([compound(element) for element in obj])
    elif isinstance(obj, tuple):
        return _Tuple([compound(element) for element in obj])
    elif isinstance(obj, dict):
        keys = [compound(key) for key in obj.keys()]
        values = [compound(value) for value in obj.values()]
        return _Dict(dict(zip(keys, values)))
    else:
        return asfunction(obj)

class Sum(OpFunction):
    """
    Abstract sum Function.
    """
    
    def __init__(self, one, two):
        
        self.one = asfunction(one)
        self.two = asfunction(two)

    def __call__(self, *args, **kwargs):
        
        return self.one(*args, **kwargs) + self.two(*args, **kwargs)

class Product(OpFunction):
    """
    Abstract product Function.
    """
    
    def __init__(self, one, two):
        
        self.one = asfunction(one)
        self.two = asfunction(two)

    def __call__(self, *args, **kwargs):
        
        return self.one(*args, **kwargs) * self.two(*args, **kwargs)

class Quotient(OpFunction):
    """
    Abstract quotient Function.
    """
    
    def __init__(self, one, two):
        
        self.one = asfunction(one)
        self.two = asfunction(two)

    def __call__(self, *args, **kwargs):
        
        return self.one(*args, **kwargs) / self.two(*args, **kwargs)

class Cmp(OpFunction):
    """
    Abstract comparison Function.
    """

    def __init__(self, A, B):
        
        self.A = asfunction(A)
        self.B = asfunction(B)

    def __call__(self, *args, **kwargs):
        
        return cmp(self.A(*args, **kwargs), self.B(*args, **kwargs))

class Less(OpFunction):
    """
    Abstract comparison Function.
    """

    def __init__(self, A, B):
        
        self.A = asfunction(A)
        self.B = asfunction(B)

    def __call__(self, *args, **kwargs):
        
        return self.A(*args, **kwargs) < self.B(*args, **kwargs)

class Greater(OpFunction):
    """
    Abstract comparison Function.
    """

    def __init__(self, A, B):
        
        self.A = asfunction(A)
        self.B = asfunction(B)

    def __call__(self, *args, **kwargs):
        
        return self.A(*args, **kwargs) > self.B(*args, **kwargs)

class LessEqual(OpFunction):
    """
    Abstract comparison Function.
    """

    def __init__(self, A, B):
        
        self.A = asfunction(A)
        self.B = asfunction(B)

    def __call__(self, *args, **kwargs):
        
        return self.A(*args, **kwargs) <= self.B(*args, **kwargs)

class GreaterEqual(OpFunction):
    """
    Abstract comparison Function.
    """

    def __init__(self, A, B):
        
        self.A = asfunction(A)
        self.B = asfunction(B)

    def __call__(self, *args, **kwargs):
        
        return self.A(*args, **kwargs) >= self.B(*args, **kwargs)

class Equal(OpFunction):
    """
    Abstract comparison Function.
    """

    def __init__(self, A, B):
        
        self.A = asfunction(A)
        self.B = asfunction(B)

    def __call__(self, *args, **kwargs):
        
        return self.A(*args, **kwargs) == self.B(*args, **kwargs)

class NotEqual(OpFunction):
    """
    Abstract comparison Function.
    """

    def __init__(self, A, B):
        
        self.A = asfunction(A)
        self.B = asfunction(B)

    def __call__(self, *args, **kwargs):
        
        return self.A(*args, **kwargs) != self.B(*args, **kwargs)

class InBetween(OpFunction):
    """
    A <= x < B

    Because *low* and *high* are likely to be non-static, the aim cannot be
    reached via subclassing :class:`Apply`.

    If numpy is available, ``numpy.logical_and`` is used to concatenate the
    outcome of ``A <= x`` and ``x < B``, else Python ``A <= x < B`` is used.
    """

    def __init__(self, value, low, high):
        
        self.value = asfunction(value)
        self.low = asfunction(low)
        self.high = asfunction(high)

    def __call__(self, *args, **kwargs):
        
        low = self.low(*args, **kwargs)
        high = self.high(*args, **kwargs)
        value = self.value(*args, **kwargs)

        if numpy_available:
            return numpy.logical_and(low <= value, value < high)
        else:
            return low <= value < high

class Power(OpFunction):
    """
    Abstract power Function.  Uses always just ``base ** exponent`` syntax,
    this might translate into numpy as well as Python exponentiation.
    """
    
    def __init__(self, base, exponent):
        
        self.base = asfunction(base)
        self.exponent = asfunction(exponent)

    def __call__(self, *args, **kwargs):
        
        return self.base(*args, **kwargs) ** self.exponent(*args, **kwargs)

class Neg(Apply):
    """
    Abstract Apply for negative calculation.
    """
    
    def __init__(self):
        
        Apply.__init__(self)

    def __call__(self, invertible):
        
        return -invertible

class Not(Apply):
    """
    Abstract Apply for negation.  If numpy is available, 
    ``numpy.logical_not()`` is used, else ``not X``.  This might be a bit
    slower for scalar input if numpy is available, but it saves us the extra 
    class ``NumpyNot``.
    """

    def __init__(self):
        
        Apply.__init__(self)

    def __call__(self, invertible):
        
        if numpy_available:
            return numpy.logical_not(invertible)
        else:
            return not invertible

class Cos(Apply):
    """Takes the cosine."""

    def __call__(self, angle):
        """If numpy is available, calculates the cosine of *angle* using 
        ``numpy.cos``.  Else, uses ``math.cos``."""

        if numpy_available:
            return numpy.cos(angle, *self.args, **self.kwargs)
        else:
            return math.cos(angle)

class Sin(Apply):
    """Takes the sine."""

    def __call__(self, angle):
        """If numpy is available, calculates the sine of *angle* using 
        ``numpy.sin``.  Else, uses ``math.sin``."""
        
        if numpy_available:
            return numpy.sin(angle, *self.args, **self.kwargs)
        else:
            return math.sin(angle)

class Exp(Apply):
    """Exponentiates."""

    def __call__(self, exponent):
        """Calculates ``exp()`` of *exponent*."""

        if numpy_available:
            return numpy.exp(exponent, *self.args, **self.kwargs)
        else:
            return math.exp(exponent)

class Sqrt(Apply):
    """Square root."""

    def __call__(self, radicand):
        """Takes the square root.  Uses numpy if available, else 
        ``math.sqrt``."""

        if numpy_available:
            return numpy.sqrt(radicand, *self.args, **self.kwargs)
        else:
            return math.sqrt(radicand)

class SumCall(Apply):
    """Calles ``.sum()`` with predefined arguments.  This is intended for
    use with ndarray-values Functions."""
    
    def __call__(self, array):
        """If numpy is available, Calls *array.sum(...)* with ``...`` being 
        the arguments handed over to *self.__init__()*.  Else, if numpy is
        not available, calls ``sum(array)`` (Python ``sum``)."""

        if numpy_available:
            return array.sum(*self.args, **self.kwargs)
        else:
            return sum(array)

class Indexing(Apply):
    """
    Returns an item from some value using a static key.

    Applications:

    1)  Indexing of a tuple with an integer.
    2)  Indexing of a ndarray with any index accepted by the ndarray.
    3)  Indexing of a dictionary.

    Example::
        
        compound([A, B]) | Indexing(0)
    """
    
    def __call__(self, indexable):
        """Indexes the value of *indexable* by calling its ``__getitem__()``
        method."""

        return indexable.__getitem__(*self.args, **self.kwargs)

class Attribute(Apply):
    """
    Returns a static attribute from the argument.
    """

    def __call__(self, host):
        """Calls ``getattr()`` on *host*."""

        return getattr(host, *self.args, **self.kwargs)

class AsType(Apply):
    """
    An :class:`Apply` derivate best to be used like this:
        
        array_like | AsType(dtype=numpy.int)
    """
    
    def __call__(self, argument):
        """Calls ``argument.astype(...)`` with ``...`` being the things
        handed over to *self.__init__()*."""

        return argument.astype(*self.args, **self.kwargs)

class Clip(OpFunction):
    """
    Clips the leaf Function's value.

    This cannot be done with an :class:`Apply`, since the clip values
    are likely to be non-static.
    """

    def __init__(self, low, high, leaf=None):
        """*low* is the Function giving the lower boundary, *high* gives the
        upper boundary, and *leaf* is the Function to be clipped."""

        self.low = asfunction(low)
        self.high = asfunction(high)
        self.leaf = asfunction(leaf)

    def __call__(self, *args, **kwargs):
        """Calls ``.low()``, ``.high()``, ``.leaf()``, and clips using the
        values.  If numpy is available, ``numpy.clip`` is used, else a Python
        logic is used."""

        low = self.low(*args, **kwargs)
        high = self.high(*args, **kwargs)
        leaf = self.leaf(*args, **kwargs)
        
        if numpy_available:
            return numpy.clip(leaf, low, high)
        else:
            if leaf < low:
                return low
            elif leaf < high:
                return leaf
            else:
                return high

class Int(OpFunction):
    """Converts via ``int()``.  Intended use case::
    
        something | Int()
        -or-
        int(something)
    """

    def __call__(self, convertible):
        """Converts *convertible* via ``int()``."""

        return int(convertible)

class Float(OpFunction):
    """Converts via ``float()``.  Intended usecase::
    
        something | Float()
        -or-
        float(something)
    """

    def __call__(self, convertible):
        """Converts *convertible* via ``float()``."""

        return float(convertible)

class Bool(OpFunction):
    """Converts via ``bool()``.  Inteded usecase::

        something | Bool()
        -or-
        bool(something)
    """
    
    def __call__(self, convertible):
        """Converts *convertible* via ``bool()``."""

        return bool(convertible)
