# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
OpenAI Agents instrumentation supporting `openai` in agent frameworks, it can be enabled by
using ``OpenAIAgentsInstrumentor``.

.. _openai: https://pypi.org/project/openai/

Usage
-----

.. code:: python

    from openai import OpenAI
    from opentelemetry.instrumentation.openai_agents import OpenAIAgentsInstrumentor

    OpenAIAgentsInstrumentor().instrument()

    # Your OpenAI agents code here
    client = OpenAI()

API
---

The OpenAI Agents instrumentation captures spans for OpenAI API calls made within agent frameworks.
It provides detailed tracing information including:

- Request and response content (if configured)
- Token usage metrics
- Model information
- Error handling

Configuration
-------------

The following environment variables can be used to configure the instrumentation:

- ``OTEL_INSTRUMENTATION_OPENAI_AGENTS_CAPTURE_CONTENT``: Set to ``true`` to capture the content of the requests and responses. Default is ``false``.
- ``OTEL_INSTRUMENTATION_OPENAI_AGENTS_CAPTURE_METRICS``: Set to ``true`` to capture metrics. Default is ``true``.
"""

import logging
from os import environ
from typing import Collection

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.openai_agents.package import _instruments
from opentelemetry.instrumentation.openai_agents.version import __version__
from opentelemetry.instrumentation.openai_agents.utils import is_content_enabled
from opentelemetry.instrumentation.openai_agents.genai_semantic_processor import GenAISemanticProcessor
from opentelemetry.semconv.schemas import Schemas
from opentelemetry.trace import get_tracer
from opentelemetry._events import get_event_logger
from agents.tracing import add_trace_processor


logger = logging.getLogger(__name__)


class OpenAIAgentsInstrumentor(BaseInstrumentor):
    """An instrumentor for OpenAI Agents frameworks."""

    def _instrument(self, **kwargs):
        """Instruments the OpenAI library for agent frameworks."""
        # TODO: Implement your instrumentation logic here
        # This is a stub implementation - add your custom instrumentation code
        
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(
            __name__,
            "",
            tracer_provider,
            schema_url=Schemas.V1_28_0.value,
        )
        event_logger_provider = kwargs.get("event_logger_provider")
        event_logger = get_event_logger(
            __name__,
            "",
            schema_url=Schemas.V1_28_0.value,
            event_logger_provider=event_logger_provider,
        )

        add_trace_processor(GenAISemanticProcessor(tracer=tracer, event_logger=event_logger))
        


    def _uninstrument(self, **kwargs):
        """Uninstruments the OpenAI library for agent frameworks."""
        pass

    def instrumentation_dependencies(self) -> Collection[str]:
        """Return a list of python packages with versions that the will be instrumented."""
        return _instruments
