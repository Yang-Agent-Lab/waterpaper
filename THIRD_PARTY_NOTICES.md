# Third-party sources

The public snapshot preserves research adapters and their source provenance.

| Component | Source and treatment |
| --- | --- |
| ESDF / heat-cold-wave adaptation | `src/esdf_flood/` retains the EDL@HKU upstream MIT notice in [UPSTREAM_LICENSE.txt](src/esdf_flood/UPSTREAM_LICENSE.txt). See that module's README for the adaptation history. |
| DiffPTS | Author repository: https://github.com/wwy155/DiffPTS, fixed commit `e9a7d1d56bd40b4bcfced29b430f6c5be6f2a424`. The full upstream checkout is fetched separately and is not redistributed here. |
| PPM | Author repository: https://github.com/ljl8336/PPM, fixed commit `6b4206a8c6bd3d298d8bdc58ebc88e5f5f6c13f9`. The full upstream checkout is fetched separately and is not redistributed here. |
| NeuralForecast, torch-timeseries and other packages | Installed separately through the dependency files; their original licenses and notices continue to apply. |

This publication does not add a blanket license for the project's own research code. Existing third-party copyright and license notices are retained. Further redistribution of upstream components should preserve their respective notices.
