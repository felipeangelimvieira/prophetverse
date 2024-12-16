# Marketing Mix Modeling (MMM)

Marketing Mix Modeling (MMM) is a statistical analysis technique that helps obtaining
insights and planning marketing strategies. It is tighly related to Time Series Analysis -
we can think of MMM as a special case of Time Series forecasting, where the goal is to
understand the incrementality of different exogenous variables on the target variable.

When Prophetverse was created, the objective was to provide a more up-to-date implementation
of Facebook's Prophet model, and add some features and customization options that were
not available in the original implementation. However, as the library evolved, it became
clear that it could be used for more than just forecasting, and that it could be a powerful
tool for MMM.

Prophetverse has the following features that make it a great choice for MMM:

* __Modularity__: Prophetverse allows one to create additive bayesian models in a modular
fashion. Users can easily use different effects to account different relationships
between the exogenous variables and the target variable, and even create their own effects.

* __Versatility__: The effects API can be used for not only adding new components,
but also adding new likelihood terms, such as the ones used for lift test.


The following effects may be of interest if you are working on MMM:

* [**GeometricAdstockEffect**](/reference/effects/#prophetverse.effects.GeometricAdstockEffect): The geometric adstock effect is a widely used technique in MMM to account
for the lagged effect of advertising on sales. It is based on the idea that the effect
of an ad on sales decays over time, and that the decay follows a geometric progression.

* [**HillEffect**](/reference/effects/#prophetverse.effects.HillEffect): The hill curve accounts for diminishing returns in the effect of an
exogenous variable on the target variable.

* [**ChainedEffects**](/reference/effects/#prophetverse.effects.ChainedEffects): The chained effect is a way to combine multiple effects into a single
one. For example, you can use adstock and hill together.

* [**LiftExperimentLikelihood**](/reference/effects/#prophetverse.effects.LiftExperimentLikelihood): The lift experiment likelihood is a likelihood term that can be
used to account for the effect of a lift test on the target variable. It is useful if you
want to understand the incrementality of a variable, and you have already run a lift test
to understand how variations in the input affect the output.

* [**ExactLikelihood**](/reference/effects/#prophetverse.effects.ExactLikelihood):
 The exact likelihood is a likelihood term that can be used to
use a reference value as the incrementality of an exogenous variable. It is useful
if another team in your company has already calculated the incrementality of a
variable and wants to use it in your MMM model.

## Related libraries

I invite you to also check other libraries for MMM. Two of them are:

* [PyMC-Marketing](https://www.pymc-marketing.io/en/stable/index.html): this is an
amazing work by PyMC's developers. It is a library that provides a set of tools for
building Bayesian models for marketing analytics. The documentation is very complete
and it's a good source of information.

* [Lightweight-MMM](https://lightweight-mmm.readthedocs.io/en/latest/index.html): this
was a library, as far as I know, created by google developers, based on Numpro. Now,
they are developing a new one, called [Meridian](https://developers.google.com/meridian?hl=fr).


## Future work

In future releases of Prophetverse, we aim at providing more tools for MMM. 
Particularly, a modular interface for running budget optimization in MMM models.
If you are interested in other features, please let us know by opening an issue in the
[repository](https://github.com/felipeangelimvieira/prophetverse)






