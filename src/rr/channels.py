import bisect
import contextlib
import inspect
import logging

from rr import pretty

__version__ = "0.1.0"

log = logging.getLogger(__name__).debug
_active_signals = []  # active primitive stack for **signals only**
_active_listeners = []  # active primitive stack for **listeners only**


def active_signal(depth=0):
    """Get the `Signal` object at the given `depth` of the active signals stack."""
    return _active_signals[-1-depth]


def active_listener(depth=0):
    """Get the `Listener` object at the given `depth` of the active listeners stack."""
    return _active_listeners[-1-depth]


@contextlib.contextmanager
def _activating(obj):
    """Context manager that inserts a primitive in the respective stack, logs its activation,
    and removes it from the respective "active stack" at the end of the `with` block.
    """
    if isinstance(obj, Signal):
        active_stack = _active_signals
    elif isinstance(obj, Listener):
        active_stack = _active_listeners
    else:
        raise TypeError("unknown primitive type {} ({})".format(type(obj), obj))
    active_stack.append(obj)
    n = len(_active_signals) + len(_active_listeners)
    log("{:>3} | {}{}".format(n, "  "*(n-1), obj))
    try:
        yield obj
    finally:
        active_stack.pop()


def _get_caller(depth=0):
    """Get the caller of this this function's caller. The`depth` in the call stack."""
    frame = inspect.currentframe()
    try:
        for _ in range(depth+2):
            frame = frame.f_back
        func_name = frame.f_code.co_name
        return frame.f_globals[func_name]
    except Exception:
        return None


def _get_caller_name(depth=0):
    """Return the name of the caller of this function's caller, or `None` if unable to retrieve
    the calling function.
    """
    func = _get_caller(depth+1)
    return None if func is None else func.__qualname__


class PriorityList(list):
    """Auxiliary list that keeps its elements sorted by their `priority` attribute."""

    __slots__ = ("priorities",)

    def __init__(self, elems=()):
        list.__init__(self)
        self.priorities = []
        for elem in elems:
            self.insert(elem)

    def insert(self, elem):
        priorities = self.priorities
        priority = elem.priority
        index = bisect.bisect_right(priorities, priority)
        priorities.insert(index, priority)
        list.insert(self, index, elem)

    def remove(self, elem):
        priorities = self.priorities
        priority = elem.priority
        index = bisect.bisect_left(priorities, priority)
        for i in range(index, len(self)):
            # Since the list is ordered by priority, we can stop looking as soon as we find an
            # item with a different priority.
            if priority != priorities[i]:
                break
            # The i-th item has the same priority as `elem`, so now we must compare the actual
            # items to make sure we're removing the correct element and not just any element with
            #  the same priority.
            if elem == self[i]:
                del priorities[i]
                del self[i]
                return
        raise ValueError("element not found {!r}".format(elem))


@pretty.klass
class Channel:
    """This class implements a channel for asynchronous communication using `Signal` and
    `Listener` primitives. Users should not use the primitive classes directly. Instead,
    the `listen()` and `emit()` methods should be used.

    Channels have a `name` attribute which defines the channel hierarchy, similar to loggers in
    the `logging` standard library module. The root channel's name is the empty string,
    and the dot character is used as separator. As an example, channel "a.b" is a child of "a"
    and the root channel.

    Emitting a signal will activate all matching listeners in the origin channel. Then,
    the signal is propagated to ancestor channels up to (and including) the root channel. Signals
    are matched firstly by type, but each listener may also include a `condition()` predicate
    that inspects the signal and decides whether to trigger the listener or not.

    Example usage:

        import contextlib
        import rr.channels

        def show_active_primitives():
            print(rr.channels.active_signal(), rr.channels.active_listener())

        root = rr.channels.Channel()
        foo = rr.channels.Channel("foo")
        with root.listen(show_active_primitives):
            foo.emit("bar")
            with foo.listen(show_active_primitives, type="bar"):
                foo.emit("spam")
                foo.emit("bar")
                foo.emit("ham")
            foo.emit("bar")
        foo.emit("bar")
    """

    # Cache of channel objects currently being used (i.e. there's at least one hard reference to
    # them). The idea for this cache comes from the `logging` module, specifically from the
    # `logging.getLogger()` function. The reason for this is to allow different modules to use
    # the exact same channel without either of them explicitly importing the channel object from
    # another module.
    __instances__ = {}

    def __new__(cls, name=""):
        channel = cls.__instances__.get(name)  # try to fetch the channel from cache
        if channel is None:
            channel = super(Channel, cls).__new__(cls)  # cache miss -> create a new object
        return channel

    def __init__(self, name=""):
        cls = type(self)
        if name in cls.__instances__:
            return  # skip initialization if this channel was obtained from cache
        self.listeners = {}  # {signal_type: [Listener]}
        self.name = name  # channel name (defines channel hierarchy as in `logging`)
        self.ancestor_names = []  # names of ancestor channels (bottom up to the root channel)
        if name != "":
            # For all channels **except the root channel**, we build a list of the names of all
            # ancestor channels. The root channel is an exception because it is the only channel
            # without any ancestors.
            parts = name.split(".")
            while len(parts) > 0:
                parts.pop()
                self.ancestor_names.append(".".join(parts))
        cls.__instances__[name] = self  # store the channel in the class' cache

    def __info__(self):
        return self.name

    @classmethod
    def clear_all(cls):
        """Calls the `clear()` method on all channels cached by `cls`."""
        for channel in cls.__instances__.values():
            channel.clear()
        cls.__instances__.clear()

    def clear(self):
        """Stop and remove all listeners from the channel."""
        for listeners in self.listeners.values():
            for listener in list(listeners):
                listener.stop()
        assert len(self.listeners) == 0

    def listen(self, callback, type=None, condition=None, priority=0.0, owner=None):
        """Creates and starts a Listener object on this channel. Returns the new Listener."""
        if owner is None:
            owner = _get_caller_name()
        listener = Listener(
            channel=self,
            callback=callback,
            type=type,
            condition=condition,
            priority=priority,
            owner=owner,
        )
        listener.start()
        return listener

    def emit(self, type, data=None, owner=None):
        """Creates and emits a Signal object on this channel. Returns the new Signal."""
        if owner is None:
            owner = _get_caller_name()
        signal = Signal(
            channel=self,
            type=type,
            data=data,
            owner=owner,
        )
        signal.emit()
        return signal

    def _insert(self, listener):
        listeners = self.listeners.get(listener.type)
        if listeners is None:
            listeners = self.listeners[listener.type] = PriorityList()
        listeners.insert(listener)

    def _remove(self, listener):
        listeners = self.listeners[listener.type]
        listeners.remove(listener)
        if len(listeners) == 0:
            del self.listeners[listener.type]

    def _emit(self, signal):
        """Find and activate all listeners attached to this channel which match `signal`,
        and then propagate the signal to ancestor channels (up to and including the root channel).
        """
        self._emit_local(signal)
        cache = type(self).__instances__
        for name in self.ancestor_names:
            ancestor = cache.get(name)
            if ancestor is not None:
                ancestor._emit_local(signal)

    def _emit_local(self, signal):
        """Emit `signal` locally on this channel. This triggers local listeners matching the
        signal.
        """
        for type in (signal.type, None):
            listeners = self.listeners.get(type)
            if listeners is None:
                continue
            for listener in list(listeners):
                listener._check(signal)


@pretty.klass
class Listener:
    """Implements a wait for a signal. Whenever a signal is emitted with the same type on the
    same channel, the listener will activate. If the type is None, the listener will be activated
    by the next signal to be emitted in the channel independently of the its type. In addition to
    filtering signals by type, listeners may also have a `condition` predicate that is able to
    inspect the contents of matching signals to further narrow down the scope of the listener and
    only activate when the desired signals are emitted.

    Listeners also have a `priority` attribute that defines the order by which listeners are
    activated when a signal triggers multiple listeners simultaneously (lower priority activate
    first). Listeners with the same priority activate by the order of their insertion into the
    channel (when the start() method is called).
    """

    def __init__(self, channel, callback, type=None, condition=None, priority=0.0, owner=None):
        self.channel = channel  # channel where the listener is deployed
        self.callback = callback  # callable executed when the listener is triggered
        self.type = type  # type of signal that triggers the listener
        self.condition = condition  # predicate used to filter matching signals
        self.priority = priority  # listener activation priority (lower goes first)
        self.owner = owner  # owner object (optional usage, can be anything)
        self.deployed = False  # True iff currently listening

    def __info__(self):
        type = self.type or "*"
        channel = self.channel.name or "."
        callback = f", cb={self.callback.__name__}"
        condition = "" if self.condition is None else f", c={self.condition.__name__}"
        priority = "" if self.priority == 0.0 else f", p={self.priority!r}"
        owner = "" if self.owner is None else f", o={self.owner!r}"
        return f"{type}@{channel}{callback}{condition}{priority}{owner}"

    def __enter__(self):
        if not self.deployed:
            self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.deployed:
            self.stop()

    def start(self):
        if not self.deployed:
            self.channel._insert(self)
            self.deployed = True

    def stop(self):
        if self.deployed:
            self.channel._remove(self)
            self.deployed = False

    def _check(self, signal):
        if self.deployed and (self.condition is None or self.condition(signal)):
            with _activating(self):
                self.callback()


@pretty.klass
class Signal:
    """Signals allow synchronizing several objects listening on the same channel. Signals
    activate listeners, which wait until a signal of matching type is emitted on the same channel.
    """

    def __init__(self, channel, type, data=None, owner=None):
        if type is None:
            raise ValueError("signal type cannot be None")
        self.channel = channel  # Channel to which the signal is associated
        self.type = type  # signal type (usually a string)
        self.data = data  # data/payload can be any Python object
        self.owner = owner  # owner object (optional usage, can be anything)

    def __info__(self):
        channel = self.channel.name or "."
        data = "" if self.data is None else f", d={self.data!r}"
        owner = "" if self.owner is None else f", o={self.owner!r}"
        return f"{self.type}@{channel}{data}{owner}"

    def emit(self):
        with _activating(self):
            self.channel._emit(self)
