import inspect
from functools import wraps
from typing import Callable


def make_partial_function(original_func: Callable, fixed_values: dict) -> Callable:
    sig = inspect.signature(original_func)

    @wraps(original_func)
    def wrapper(*args, **kwargs):
        # Start with fixed values
        bound_args = dict(fixed_values)

        # Bind positional/keyword args from caller
        for name, value in zip(remaining_params, args):
            bound_args[name] = value
        bound_args.update(kwargs)

        return original_func(**bound_args)

    # Only keep parameters NOT in fixed_values
    remaining_params = [name for name in sig.parameters if name not in fixed_values]
    new_params = [sig.parameters[name] for name in remaining_params]

    # Set the new __signature__ for introspection
    wrapper.__signature__ = sig.replace(parameters=new_params)  # type:ignore

    return wrapper


def make_routed_function(
    original_func: Callable, param_name: str, resolver: Callable[[], object]
) -> Callable:
    """Like ``make_partial_function`` but the removed parameter's value is
    computed PER CALL by ``resolver`` instead of fixed at registration.

    This is what lets one shared server route a tool to a different collection
    per connection: the tool's ``collection_name`` is dropped from the exposed
    signature, and resolved at request time (e.g. from a per-connection header)
    when the tool is actually invoked.
    """
    sig = inspect.signature(original_func)
    remaining_params = [name for name in sig.parameters if name != param_name]

    @wraps(original_func)
    def wrapper(*args, **kwargs):
        bound_args = {}
        for name, value in zip(remaining_params, args):
            bound_args[name] = value
        bound_args.update(kwargs)
        bound_args[param_name] = resolver()
        return original_func(**bound_args)

    new_params = [sig.parameters[name] for name in remaining_params]
    wrapper.__signature__ = sig.replace(parameters=new_params)  # type:ignore

    return wrapper
