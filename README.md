# mm-toolbox
toolbox of fast mm-related funcs

## Installation

To install `mm_toolbox`, follow these steps:

1. _Optional: If you are familiar with virtual environments, create one now and activate it. If not, this step is not necessary:_

```console
$ virtualenv venv
$ source venv/bin/activate
```

2. Clone the repository or download the source code to your local machine.
   
3. With your virtual environment activated, navigate to the root directory of `mm_toolbox` (where `setup.py` is located) and run:
    ```bash
    python setup.py install
    ```

This will install `mm_toolbox` and its dependencies into your virtual environment.

## Usage

After installing `mm_toolbox`, you can start using it in your projects by importing the necessary modules and functions. Here's an example:

```python
from mm_toolbox.orderbook.orderbook import Orderbook

# Example usage of the orderbook from mm_toolbox
base_orderbook = Orderbook(size=500)
```

## Contributing

Please create [issues](https://github.com/beatzxbt/mm-toolbox/issues) to flag bugs or suggest new features and feel free to create a [pull request](https://github.com/beatzxbt/mm-toolbox/pulls) with any improvements.

## License

`mm_toolbox` is licensed under the MIT License. See the [LICENSE](LICENSE) file in the repository for more details.

## Performance tricks for Numba

1. Look at https://numba.pydata.org/numba-doc/dev/reference/envvars.html
  - Set NUMBA_OPT: max
  - Set NUMBA_ENABLE_AVX: 1

(Will add more later)

### Contact 

If you have any questions/suggestions regarding the repository, or just want to have a chat, my handles are below 👇🏼

Twitter: [@beatzXBT](https://twitter.com/BeatzXBT) | Discord: gamingbeatz
