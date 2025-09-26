Compile Time Caching Configuration
=========================================================
**Authors:** `Oguz Ulgen <https://github.com/oulgen>`_ and `Sam Larsen <https://github.com/masnesral>`_

Introduction
------------------

PyTorch Compiler implements several caches to reduce compilation latency.
This recipe demonstrates how you can configure various parts of the caching in ``torch.compile``.
PyTorch 컴파일러는 컴파일 지연을 줄이기 위해 여러 캐시를 구현하고 있습니다.
이 예제는 ``torch.compile`` 캐시의 다양한 부분을 어떻게 구성할 수 있는지 보여주고 있습니다.

Prerequisites
-------------------

Before starting this recipe, make sure that you have the following:
이 예제를 시작하기 전, 다음이 준비되어있는지 확인해주세요.

* Basic understanding of ``torch.compile``. See:

  * `torch.compiler API documentation <https://pytorch.org/docs/stable/torch.compiler.html#torch-compiler>`__
  * `Introduction to torch.compile <https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html>`__
  * `Compile Time Caching in torch.compile <https://pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html>`__

* PyTorch 2.4 or later

Inductor Cache Settings
----------------------------

Most of these caches are in-memory, only used within the same process, and are transparent to the user. An exception is caches that store compiled FX graphs (``FXGraphCache``, ``AOTAutogradCache``). These caches allow Inductor to avoid recompilation across process boundaries when it encounters the same graph with the same Tensor input shapes (and the same configuration). The default implementation stores compiled artifacts in the system temp directory. An optional feature also supports sharing those artifacts within a cluster by storing them in a Redis database.
캐시 대부분은 인메모리 내에 존재하며, 동일한 프로세스 상에서만 사용되고, 사용자에게 투명하게 동작합니다.
예외는 컴파일된 FX graph (``FXGraphCache``, ``AOTAutogradCache``)를 저장하는 캐시입니다.
이러한 캐시들은 인덕터가 같은 텐서 입력 형태를 가진 동일한 그래프를 마주쳤을 때, 프로세스와 상관없이 다시 컴파일하지 않도록 합니다.
기본 구현체는 컴파일된 아티팩트를 시스템 내의 임시 디렉터리에 저장합니다. 선택적 기능으로, Redis 데이터베이스에 아티팩트를 저장하여 클러스터 내에서 공유할 수 있습니다.

There are a few settings relevant to caching and to FX graph caching in particular.
특히 FX graph 캐싱과 관련된 몇 가지 설정이 있습니다.

The settings are accessible via environment variables listed below or can be hard-coded in the Inductor’s config file.
이 설정들은 인덕터의 설정 파일에 하드 코딩하거나, 아래 나열된 환경 변수를 통해 접근할 수 있습니다.

TORCHINDUCTOR_FX_GRAPH_CACHE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This setting enables the local FX graph cache feature, which stores artifacts in the host’s temp directory. Setting it to ``1`` enables the feature while any other value disables it. By default, the disk location is per username, but users can enable sharing across usernames by specifying ``TORCHINDUCTOR_CACHE_DIR`` (below).
이 설정은 호스트의 임시 디렉토리에 아티팩트를 저장하는 로컬 FX graph 캐시 기능을 활성화합니다.
값을 1로 설정하면 기능이 활성화되고, 그 이외의 값으로 설정하면 기능이 비활성화 됩니다.
기본적으로 디스크 위치는 각 사용자 이름으로 설정되지만, 사용자가 아래의 ``TOCHINDUCTOR_CACHE_DIR``을 지정하면 사용자 간 공유도 가능합니다.

TORCHINDUCTOR_AUTOGRAD_CACHE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This setting extends ``FXGraphCache`` to store cached results at the ``AOTAutograd`` level, rather than at the Inductor level. Setting it to ``1`` enables this feature, while any other value disables it.
By default, the disk location is per username, but users can enable sharing across usernames by specifying ``TORCHINDUCTOR_CACHE_DIR`` (below).
``TORCHINDUCTOR_AUTOGRAD_CACHE`` requires ``TORCHINDUCTOR_FX_GRAPH_CACHE`` to work. The same cache dir stores cache entries for ``AOTAutogradCache`` (under ``{TORCHINDUCTOR_CACHE_DIR}/aotautograd``) and ``FXGraphCache`` (under ``{TORCHINDUCTOR_CACHE_DIR}/fxgraph``).
이 설정은 ``FXGraphCache``를 확장하여 캐시된 결과를 인덕터 레벨이 아니라 ``AOTAutograd`` 레벨에 저장하도록 합니다.
값을 1로 설정하면 기능이 활성화되고, 그 이외의 값으로 설정하면 기능이 비활성화 됩니다.
기본적으로, 디스크 위치는 각 사용자 이름별로 설정되지만, 사용자가 아래의 ``TORCHINDUCTOR_CACHE_DIR``을 지정하면 사용자 간 공유도 가능합니다.
``TORCHINDUCTOR_AUTOGRAD_CACHE`` 기능을 사용하려면 ``TORCHINDUCTOR_FX_GRAPH_CACHE``가 활성화되어 있어야 합니다.


TORCHINDUCTOR_CACHE_DIR
~~~~~~~~~~~~~~~~~~~~~~~~
This setting specifies the location of all on-disk caches. By default, the location is in the system temp directory under ``torchinductor_<username>``, for example, ``/tmp/torchinductor_myusername``.
이 설정은 모든 디스크 캐시의 위치를 지정합니다. 기본적으로 시스템 임시 디렉터리 내의 ``torchinductor_<username>`` 폴더에 저장되며, 예시로 ``/tmp/torchinductor_myusername`` 처럼 사용됩니다.

Note that if ``TRITON_CACHE_DIR`` is not set in the environment, Inductor sets the ``Triton`` cache directory to this same temp location, under the Triton sub-directory.
환경 변수 ``TRITON_CACHE_DIR``가 설정되어있지 않으면, 인덕터는 ``Triton`` 캐시 디렉터리를 동일한 임시 디렉터리 아래에 Triton 서브 디렉터리로 설정합니다.

TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This setting enables the remote FX graph cache feature. The current implementation uses ``Redis``. ``1`` enables caching, and any other value disables it. The following environment variables configure the host and port of the Redis server:
이 설정은 원격으로 FX graph 캐싱 기능을 활성화합니다. 현재 구현은 ``Redis``를 사용합니다. 값이 1이면 캐시가 활성화되고, 그 이외의 값으로 설정하면 기능이 비활성화 됩니다. Redis 서버의 호스트와 포트는 다음 환경 변수를 통해 설정할 수 있습니다.

``TORCHINDUCTOR_REDIS_HOST`` (defaults to ``localhost``)
``TORCHINDUCTOR_REDIS_PORT`` (defaults to ``6379``)

.. note::

    Note that if Inductor locates a remote cache entry, it stores the compiled artifact in the local on-disk cache; that local artifact would be served on subsequent runs on the same machine.
만약 인덕터가 원격 캐시 항목을 찾으면, 컴파일된 아티팩트를 로컬 디스크 캐시에 저장합니다, 동일한 머신에서 실행될 때는 이 로컬 캐시가 사용됩니다.

TORCHINDUCTOR_AUTOGRAD_REMOTE_CACHE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Similar to ``TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE``, this setting enables the remote ``AOTAutogradCache`` feature. The current implementation uses Redis. Setting it to ``1`` enables caching, while any other value disables it. The following environment variables are used to configure the host and port of the ``Redis`` server:
* ``TORCHINDUCTOR_REDIS_HOST`` (defaults to ``localhost``)
* ``TORCHINDUCTOR_REDIS_PORT`` (defaults to ``6379``)

`TORCHINDUCTOR_AUTOGRAD_REMOTE_CACHE`` requires ``TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE`` to be enabled in order to function. The same Redis server can be used to store both AOTAutograd and FXGraph cache results.

TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This setting enables a remote cache for ``TorchInductor``’s autotuner. Similar to remote FX graph cache, the current implementation uses Redis. Setting it to ``1`` enables caching, while any other value disables it. The same host / port environment variables mentioned above apply to this cache.

TORCHINDUCTOR_FORCE_DISABLE_CACHES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Set this value to ``1`` to disable all Inductor caching. This setting is useful for tasks like experimenting with cold-start compile times or forcing recompilation for debugging purposes.

Conclusion
-------------
In this recipe, we have learned how to configure PyTorch Compiler's caching mechanisms. Additionally, we explored the various settings and environment variables that allow users to configure and optimize these caching features according to their specific needs.

