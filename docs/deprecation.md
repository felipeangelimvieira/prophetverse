# Deprecation policy

!!! note
    Deprecation policy of Prophetverse takes place for versions `>=0.4.0`.


## Versioning

Prophetverse follows the [Semantic Versioning](https://semver.org/) scheme, which means that the version number is composed of three parts: `MAJOR.MINOR.PATCH`. However, we do not plan to release any `MAJOR` version until
a large user base is built and the library is considered stable.

### How breaking changes in Prophetverse API are released

Breaking changes and deprecations are released in a two-step process:

1. A new minor version is released with warnings to inform the users about the
upcoming breaking changes. A `FutureWarning` is used to indicate the version in which 
the change will be efective.
2. The next minor version is released with the breaking changes and the deprecations
are removed.

For example, if we are in version 0.4.0 and a breaking change is introduced to enhance
user experience, version 0.5.0 will keep the behaviour of 0.4.0 but will be released
with a `FutureWarning`. 0.6.0 will be released with the breaking change and the warning
will be removed.

We will try to follow this policy as much as possible, but there may be cases where
we need to make exceptions. In any case, we will always try to minimize the impact on
the users. The larger the user base, the more we will try to avoid breaking changes.
If any change affects your code, please let us know and we will do our best to help you.