# Unit Testing Guidelines

### Guidelines

In general, all functional code should be tested with unit tests. All unit tests should comply with the following requirements:

1. Unit tests should use the [pytest ](https://docs.pytest.org/en/7.1.x/)module.
2. The tests that cover a module called `pyreal/path/to/a_module.py` should be implemented in a separated module called `tests/pyreal/path/to/test_a_module.py`. Note that the module name has the `test_` prefix and is located in a path similar to the one of the tested module, just inside the `tests` folder.
3. Each method of the tested module should have at least one associated test method, and each test method should cover only **one** use case or scenario.
4. Test case methods should start with the `test_` prefix and have descriptive names that indicate which scenario they cover. Names such as `test_some_methed_input_none`, `test_some_method_value_error` or `test_some_method_timeout` are right, but names like `test_some_method_1`, `some_method` or `test_error` are not.
5. Each test should validate only what the code of the method being tested does, and not cover the behavior of any third party package or tool being used, which is assumed to work properly as far as it is being passed the right values.
6. Any third party tool that may have any kind of random behavior, such as some Machine Learning models, databases or Web APIs, can be mocked using the `mock` library, and the only thing that will be tested is that our code passes the right values to them.
7. Unit tests should not use anything from outside the test and the code being tested. This includes not reading or writing to any file system or database, which will be properly mocked.

