from skbase.base import BaseObject


class BaseOptimizationObjective(BaseObject):

    _tags = {"backend": None, "name": None}

    def __call__(self, model, X, horizon, columns):
        raise NotImplementedError(
            "Utility function must be callable with model, X, period and columns"
        )


class BaseConstraint(BaseObject):

    _tags = {"backend": None, "name": None}

    def __call__(self, model, X, horizon, columns):
        raise NotImplementedError(
            "Constraint function must be callable with model, X, period and columns"
        )


class BaseDecisionVariableTransform(BaseObject):

    def fit(self, X, horizon, columns):
        """Fit the decision variable to the data"""
        self.horizon_ = horizon
        self.columns_ = columns
        return self._fit(X, horizon, columns)

    def transform(self, x):
        """Transform the decision variable to the original space"""
        return self._transform(x)

    def inverse_transform(self, xt):
        """Return the initial guess for the decision variable"""
        return self._inverse_transform(xt)

    def _fit(self, X, horizon, columns):
        pass

    def _transform(self, x):
        """Transform the decision variable to the original space"""
        raise NotImplementedError("Decision variable must implement transform method")

    def _inverse_transform(self, xt):
        """Return the initial guess for the decision variable"""
        raise NotImplementedError(
            "Decision variable must implement initial_guess method"
        )


class BaseBudgetOptimizer(BaseObject):
    """Base class for budget optimization.

    Budget optimization is an optimization of a set of input variables.
    Optimizers are meant to be used in conjunction with a model. The arguments
    should be divided as follows:

    * init arguments:
        * utility function : an specification of the objective function to be
            optimized
        * contraints: a list of constraints, such as Budget, ChannelCap, etc.
        * optimization related arguments
    * optimize arguments:
        * model
        * X : pd.DataFrame
        * period : pd.PeriodIndex
        * columns : list[str]
    """

    _tags = {"backend": None}

    def __init__(
        self, constraints: list[BaseConstraint], objective: BaseOptimizationObjective
    ):
        self.constraints = constraints
        self.objective = objective

        self._validade_init_args()
        super().__init__()

    def optimize(self, model, X, horizon, columns):
        return self._optimize(
            model=model,
            X=X,
            horizon=horizon,
            columns=columns,
        )

    def _validade_init_args(self):

        for constraint in self.constraints:
            if constraint.get_tag("backend") != self.get_tag("backend"):
                raise ValueError(
                    f"Constraint {constraint} has a different backend than the optimizer {self}"
                )

        if self.objective.get_tag("backend") != self.get_tag("backend"):
            raise ValueError(
                f"Objective {self.objective} has a different backend than the optimizer {self}"
            )


# Helper function to get all subclasses (direct and indirect)
def get_all_subclasses(cls):
    """
    Recursively retrieves all subclasses of a given class.

    Args:
        cls (type): The class for which to find subclasses.

    Returns:
        list: A list of all subclasses (descendants) of cls.
    """
    all_subclasses = set()
    # Get direct subclasses
    direct_subclasses = cls.__subclasses__()
    for subclass in direct_subclasses:
        all_subclasses.add(subclass)
        # Recursively add subclasses of these direct subclasses
        all_subclasses.update(get_all_subclasses(subclass))
    return list(all_subclasses)


# Main function to find the class by tags
def find_class_by_tags(
    name_tag_value,
    backend_tag_value,
    base_classes_to_search=(BaseOptimizationObjective, BaseConstraint),
):
    """
    Finds a child class that has the specified 'name' and 'backend' tags.

    The search is performed among the descendants of the classes provided in
    `base_classes_to_search`.

    Args:
        name_tag_value (str): The value for the 'name' tag to search for.
        backend_tag_value (str): The value for the 'backend' tag to search for.
        base_classes_to_search (type or tuple/list of types, optional):
            A single base class or an iterable (tuple or list) of base classes
            whose children will be searched. Defaults to (BaseUtility, BaseConstraint).

    Returns:
        type: The first child class found that matches the specified tags.
              Returns None if no such class is found.
    """
    search_targets = []
    if isinstance(base_classes_to_search, type):  # A single class was passed
        search_targets = [base_classes_to_search]
    elif isinstance(base_classes_to_search, (list, tuple)):
        search_targets = base_classes_to_search
    else:
        # This case should ideally raise an error or be handled more strictly,
        # but for flexibility, we'll try to iterate if possible.
        # However, it's best to pass a type or a list/tuple of types.
        try:
            search_targets = list(base_classes_to_search)
        except TypeError:
            raise TypeError(
                "base_classes_to_search must be a class or an iterable of classes."
            )

    for base_class in search_targets:
        if not isinstance(base_class, type):
            print(
                f"Warning: Item '{base_class}' in base_classes_to_search is not a class type. Skipping."
            )
            continue

        # Get all subclasses (direct and indirect) of the current base_class
        candidate_classes = get_all_subclasses(base_class)

        for child_class in candidate_classes:
            # Check if the class has a _tags attribute and it's a dictionary
            if hasattr(child_class, "_tags") and isinstance(
                getattr(child_class, "_tags"), dict
            ):
                tags = getattr(child_class, "_tags")
                # Check if the name and backend tags match
                if (
                    tags.get("name") == name_tag_value
                    and tags.get("backend") == backend_tag_value
                ):
                    return child_class  # Return the first matching class
    return None  # No matching class found
