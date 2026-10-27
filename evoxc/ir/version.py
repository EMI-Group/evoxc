"""Serialization format version for the EvoXIR payload.

The IR payload format is versioned so that every serialized artifact is
self-describing: a loader must reject any payload whose version it does not
know (never silent upgrade, never silent downgrade).

Bump ``IR_FORMAT_VERSION`` on ANY change to the wire schema (op encoding,
value/type encoding, constant encoding, hash computation). Distinct from
``EVOXC_FORMAT_VERSION`` in ``evoxc.persist``, which versions the outer
``.evoxcgraph`` container.
"""

IR_FORMAT_VERSION = 1
