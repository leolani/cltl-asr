# cltl-asr

Speech to text service for Leolani. This repository is a component of the [Leolani framework](https://github.com/leolani/cltl-combot).
For usage of the component within the framework see the instructions there.

## Description

This package contains multiple implementations to convert text from spoken language for any written text.

## Getting started

### Prerequisites

This repository uses Python >= 3.9

Be sure to run in a virtual python environment (e.g. conda, venv, mkvirtualenv, etc.)

### Installation

1. In the root directory of this repo run

    ```bash
    pip install -e .
    ```

### Implementations

There are various implementations included in `cltl.asr`. Depending on which
implementation  is used different dependencies are required for this package.
To include those during installation specify the required extra dependencies
with pip, e.g. like:

    pip install cltl.asr[whisper]

For the available options refer to `setup.py`.

#### Whisper ASR

#### Whisper C++ ASR

This implementation utilizes [whisper.cpp](https://github.com/ggerganov/whisper.cpp),
which is a C++ implementation of whisper that can be run natively and provides
improved performance over the Python implemenation of Whisper.

In order to use this implementation a *whisper.cpp* server needs to be setup.
To setup the server follow the instructions in the [whisper.cpp README](https://github.com/ggerganov/whisper.cpp)
and [server specific README](https://github.com/ggerganov/whisper.cpp/tree/master/examples/server).

For the server setup on OS X on an Apple Silicon device refer to this
[setup script](support/setup_whisper_cpp.sh).

#### Google ASR

#### Speechbrain ASR

### Usage

For using this repository as a package in a different project and on a different virtual environment, you may

- install a published version from PyPI:

    ```bash
    pip install cltl.asr
    ```

- or, for the latest snapshot, run:

    ```bash
    pip install git+git://github.com/leolani/cltl-asr.git@main
    ```

Then you can import it in a python script as:

```python
import numpy as np
import soundfile as sf
from importlib_resources import path
from cltl.asr.speechbrain_asr import SpeechbrainASR

asr = SpeechbrainASR("speechbrain/asr-transformer-transformerlm-librispeech")

with path("resources", "test.wav") as wav:
    speech_array, sampling_rate = sf.read(wav, dtype=np.int16)
transcript = asr.speech_to_text(speech_array, sampling_rate)

```

## Examples

Please take a look at the example scripts provided to get an idea on how to run and use this package. Each example has a
comment at the top of the script describing the behaviour of the script.

For these example scripts, you need

1. To change your current directory to ./examples/

1. Run some examples (e.g. python test_speechbrain_asr.py)

## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any
contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### To DO

- Fix logging
- Fix config of language, save/play audio file, audio directory
- Check if we can switch voices via different APIs
- Check implementation middle layer

## License

Distributed under the MIT License. See [`LICENSE`](https://github.com/leolani/cltl-asr/blob/main/LICENCE)
for more information.

## Authors

* [Thomas Baier](https://www.linkedin.com/in/thomas-baier-05519030/)
* [Selene Báez Santamaría](https://selbaez.github.io/)
* [Piek Vossen](https://github.com/piekvossen)


