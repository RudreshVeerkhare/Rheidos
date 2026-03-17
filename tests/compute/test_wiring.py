"""Tests for rheidos.compute.wiring module."""

import pytest
import numpy as np
from dataclasses import dataclass

from rheidos.compute import (
    ModuleBase,
    Registry,
    ResourceSpec,
    ResourceRef,
    ResourceKey,
    World,
    producer,
    producer_output,
    WiredProducer,
    out_field,
)


class TestOutField:
    """Test out_field marker function."""

    def test_out_field_default(self):
        """out_field creates field metadata."""
        field_obj = out_field()
        assert field_obj.metadata.get("io") == "out"

    def test_out_field_with_alloc(self):
        """out_field stores alloc callable."""

        def alloc_fn(reg, spec):
            return np.zeros((10,))

        field_obj = out_field(alloc=alloc_fn)
        assert field_obj.metadata.get("alloc") is alloc_fn


class TestProducerDecorator:
    """Test decorator-based producer metadata."""

    def test_producer_output_records_alloc(self):
        """producer_output stores name and alloc callable."""

        def alloc_fn(reg, ctx):
            return np.zeros((2,), dtype=np.float32)

        output = producer_output("result", alloc=alloc_fn)
        assert output.name == "result"
        assert output.alloc is alloc_fn

    def test_producer_attaches_spec(self):
        """producer attaches metadata that bind_producers can discover."""

        class DemoModule(ModuleBase):
            NAME = "demo"

            @producer(
                inputs=("a", "b"),
                outputs=("out", producer_output("cache")),
                allow_none=("b",),
                ignore=("scratch",),
            )
            def run(self, ctx):
                pass

        spec = DemoModule.run.__dict__["__rheidos_producer_spec__"]
        assert spec.inputs == ("a", "b")
        assert tuple(output.name for output in spec.outputs) == ("out", "cache")
        assert spec.allow_none == ("b",)
        assert spec.ignore == ("scratch",)


class TestDecoratedProducerRuntime:
    """Test decorator-based producer execution."""

    def test_context_ensure_outputs_uses_custom_alloc(self):
        """Decorated producers can allocate outputs through ctx.ensure_outputs()."""
        world = World()

        def alloc_out(_reg, ctx):
            return np.zeros_like(ctx.inputs.a.peek())

        class AllocModule(ModuleBase):
            NAME = "alloc"

            def __init__(self, world, **kwargs):
                super().__init__(world, **kwargs)
                self.a = self.resource(
                    "a",
                    declare=True,
                    buffer=np.array([2.0], dtype=np.float32),
                )
                self.out = self.resource("out")
                self.bind_producers()

            @producer(
                inputs=("a",),
                outputs=(producer_output("out", alloc=alloc_out),),
            )
            def fill(self, ctx):
                out = ctx.ensure_outputs(require_shape=False)["out"].peek()
                out[:] = ctx.inputs.a.get() + 3.0
                ctx.outputs.out.commit()

        module = world.require(AllocModule)
        world.reg.ensure(module.out.name)
        assert np.array_equal(module.out.peek(), np.array([5.0], dtype=np.float32))

    def test_context_commit_rejects_unknown_output(self):
        """ctx.commit rejects unknown output names."""
        world = World()

        class BadCommitModule(ModuleBase):
            NAME = "bad_commit"

            def __init__(self, world, **kwargs):
                super().__init__(world, **kwargs)
                self.a = self.resource("a", declare=True, buffer=np.array([1.0]))
                self.out = self.resource(
                    "out",
                    spec=ResourceSpec(kind="numpy", dtype=np.float64, shape=(1,)),
                )
                self.bind_producers()

            @producer(inputs=("a",), outputs=("out",))
            def run(self, ctx):
                ctx.commit(missing=np.array([1.0]))

        module = world.require(BadCommitModule)
        with pytest.raises(KeyError, match="Did you mean 'out'\\?"):
            world.reg.ensure(module.out.name)

    def test_context_inputs_attribute_typo_is_informative(self):
        """ctx.inputs attribute errors include close-name suggestions."""
        world = World()

        class TypoInputModule(ModuleBase):
            NAME = "typo_input"

            def __init__(self, world, **kwargs):
                super().__init__(world, **kwargs)
                self.mesh = self.resource(
                    "mesh",
                    declare=True,
                    buffer=np.array([1.0]),
                )
                self.out = self.resource(
                    "out",
                    spec=ResourceSpec(kind="numpy", dtype=np.float64, shape=(1,)),
                )
                self.bind_producers()

            @producer(inputs=("mesh",), outputs=("out",))
            def run(self, ctx):
                ctx.inputs.mseh.get()

        module = world.require(TypoInputModule)
        with pytest.raises(AttributeError, match="Did you mean 'mesh'\\?"):
            world.reg.ensure(module.out.name)


class TestWiredProducerIOType:
    """Test WiredProducer IO type inference."""

    def test_io_type_explicit(self):
        """WiredProducer with explicit IO_TYPE."""

        @dataclass
        class MyIO:
            output: ResourceRef = out_field()

        class MyProducer(WiredProducer):
            IO_TYPE = MyIO

            def compute(self, reg):
                pass

        assert MyProducer.IO_TYPE is MyIO

    def test_io_type_from_generic(self):
        """WiredProducer infers IO_TYPE from generic."""

        @dataclass
        class MyIO:
            output: ResourceRef = out_field()

        class MyProducer(WiredProducer[MyIO]):
            def compute(self, reg):
                pass

        assert MyProducer.IO_TYPE is MyIO


class TestWiredProducerInit:
    """Test WiredProducer initialization."""

    def test_init_with_io_object(self):
        """Initialize with IO object."""

        @dataclass
        class SimpleIO:
            out: ResourceRef = out_field()

        class SimpleProducer(WiredProducer[SimpleIO]):
            def compute(self, reg):
                pass

        registry = Registry()
        out_ref = ResourceRef(registry, ResourceKey("out"))
        io = SimpleIO(out=out_ref)
        producer = SimpleProducer(io=io)
        assert producer.io is io

    def test_init_with_kwargs(self):
        """Initialize with kwargs (requires IO_TYPE)."""

        @dataclass
        class SimpleIO:
            out: ResourceRef = out_field()

        class SimpleProducer(WiredProducer[SimpleIO]):
            def compute(self, reg):
                pass

        registry = Registry()
        out_ref = ResourceRef(registry, ResourceKey("out"))
        producer = SimpleProducer(out=out_ref)
        assert isinstance(producer.io, SimpleIO)

    def test_setup_called_after_wiring(self):
        """setup hook runs after io/inputs/outputs are wired."""

        @dataclass
        class SimpleIO:
            input: ResourceRef
            output: ResourceRef = out_field()

        class SimpleProducer(WiredProducer[SimpleIO]):
            def setup(self):
                self.setup_called = True
                assert isinstance(self.io, SimpleIO)
                assert "input" in self.inputs
                assert "output" in self.outputs

            def compute(self, reg):
                pass

        registry = Registry()
        in_ref = ResourceRef(registry, ResourceKey("input"))
        out_ref = ResourceRef(registry, ResourceKey("output"))
        producer = SimpleProducer(input=in_ref, output=out_ref)
        assert getattr(producer, "setup_called", False)

    def test_init_non_dataclass_raises(self):
        """Non-dataclass IO raises TypeError."""

        class NotDataclass:
            pass

        class BadProducer(WiredProducer):
            IO_TYPE = NotDataclass

            def compute(self, reg):
                pass

        with pytest.raises(TypeError, match="dataclass"):
            BadProducer(io=NotDataclass())

    def test_init_no_io_type_raises(self):
        """No IO_TYPE and no generic raises TypeError."""

        class NoIOType(WiredProducer):
            def compute(self, reg):
                pass

        with pytest.raises(TypeError, match="IO_TYPE"):
            NoIOType()


class TestWiredProducerWiring:
    """Test WiredProducer input/output wiring."""

    def test_outputs_inferred(self):
        """Outputs inferred from out_field()."""

        @dataclass
        class IO:
            result1: ResourceRef = out_field()
            result2: ResourceRef = out_field()

        class Prod(WiredProducer[IO]):
            def compute(self, reg):
                pass

        registry = Registry()
        ref1 = ResourceRef(registry, ResourceKey("r1"))
        ref2 = ResourceRef(registry, ResourceKey("r2"))
        io = IO(result1=ref1, result2=ref2)
        prod = Prod(io=io)
        assert "r1" in prod.outputs
        assert "r2" in prod.outputs

    def test_inputs_inferred(self):
        """Inputs inferred from ResourceRef fields."""

        @dataclass
        class IO:
            input_ref: ResourceRef
            output: ResourceRef = out_field()

        class Prod(WiredProducer[IO]):
            def compute(self, reg):
                pass

        registry = Registry()
        in_ref = ResourceRef(registry, ResourceKey("in"))
        out_ref = ResourceRef(registry, ResourceKey("out"))
        io = IO(input_ref=in_ref, output=out_ref)
        prod = Prod(io=io)
        assert "in" in prod.inputs
        assert "out" in prod.outputs

    def test_non_ref_fields_ignored(self):
        """Non-ResourceRef fields ignored."""

        @dataclass
        class IO:
            data: np.ndarray  # Not a ResourceRef
            output: ResourceRef = out_field()

        class Prod(WiredProducer[IO]):
            def compute(self, reg):
                pass

        registry = Registry()
        out_ref = ResourceRef(registry, ResourceKey("out"))
        io = IO(data=np.array([1, 2, 3]), output=out_ref)
        prod = Prod(io=io)
        assert len(prod.inputs) == 0
        assert "out" in prod.outputs


class TestWiredProducerRequireInputs:
    """Test WiredProducer.require_inputs() method."""

    def test_require_inputs_all_present(self):
        """require_inputs succeeds when all inputs have buffers."""
        registry = Registry()
        registry.declare("input1", buffer=np.array([1.0]))
        registry.declare("input2", buffer=np.array([2.0]))

        @dataclass
        class IO:
            in1: ResourceRef
            in2: ResourceRef
            output: ResourceRef = out_field()

        class Prod(WiredProducer[IO]):
            def compute(self, reg):
                pass

        in1_ref = ResourceRef(registry, ResourceKey("input1"))
        in2_ref = ResourceRef(registry, ResourceKey("input2"))
        out_ref = ResourceRef(registry, ResourceKey("output"))
        io = IO(in1=in1_ref, in2=in2_ref, output=out_ref)
        prod = Prod(io=io)

        inputs = prod.require_inputs()
        assert "in1" in inputs
        assert "in2" in inputs

    def test_require_inputs_missing_required_raises(self):
        """require_inputs raises if required input has no buffer."""
        registry = Registry()
        registry.declare("input1")  # No buffer

        @dataclass
        class IO:
            in1: ResourceRef
            output: ResourceRef = out_field()

        class Prod(WiredProducer[IO]):
            def compute(self, reg):
                pass

        in1_ref = ResourceRef(registry, ResourceKey("input1"))
        out_ref = ResourceRef(registry, ResourceKey("output"))
        io = IO(in1=in1_ref, output=out_ref)
        prod = Prod(io=io)

        # Should raise RuntimeError when buffer is None
        with pytest.raises(RuntimeError, match="missing required inputs"):
            prod.require_inputs()

    def test_require_inputs_allow_none(self):
        """require_inputs allows None buffers for specified fields."""
        registry = Registry()
        registry.declare("input1")  # No buffer

        @dataclass
        class IO:
            in1: ResourceRef
            output: ResourceRef = out_field()

        class Prod(WiredProducer[IO]):
            def compute(self, reg):
                pass

        in1_ref = ResourceRef(registry, ResourceKey("input1"))
        out_ref = ResourceRef(registry, ResourceKey("output"))
        io = IO(in1=in1_ref, output=out_ref)
        prod = Prod(io=io)

        # Should not raise with allow_none set
        inputs = prod.require_inputs(allow_none=["in1"])
        assert "in1" in inputs

    def test_require_inputs_ignore(self):
        """require_inputs ignores specified fields."""
        registry = Registry()
        # Declare both inputs with buffers
        registry.declare("input1", buffer=np.array([1.0]))
        registry.declare("input2", buffer=np.array([2.0]))

        @dataclass
        class IO:
            in1: ResourceRef
            in2: ResourceRef
            output: ResourceRef = out_field()

        class Prod(WiredProducer[IO]):
            def compute(self, reg):
                pass

        in1_ref = ResourceRef(registry, ResourceKey("input1"))
        in2_ref = ResourceRef(registry, ResourceKey("input2"))
        out_ref = ResourceRef(registry, ResourceKey("output"))
        io = IO(in1=in1_ref, in2=in2_ref, output=out_ref)
        prod = Prod(io=io)

        # in2 should be ignored (not in result, not checked)
        inputs = prod.require_inputs(ignore=["in2"])
        assert "in2" not in inputs

    def test_io_type_inference_from_generic(self):
        """IO_TYPE is inferred from WiredProducer generic parameter."""

        @dataclass
        class MyIO:
            x: ResourceRef
            y: ResourceRef = out_field()

        class MyProd(WiredProducer[MyIO]):
            def compute(self, reg):
                pass

        # IO_TYPE should be set by __init_subclass__
        assert MyProd.IO_TYPE is MyIO

    def test_wired_producer_with_kwargs(self):
        """WiredProducer can be initialized with kwargs instead of io object."""
        registry = Registry()
        registry.declare("a", buffer=np.array([1.0]))
        registry.declare("b")

        @dataclass
        class IO:
            input: ResourceRef
            result: ResourceRef = out_field()

        class Prod(WiredProducer[IO]):
            def compute(self, reg):
                pass

        a_ref = ResourceRef(registry, ResourceKey("a"))
        b_ref = ResourceRef(registry, ResourceKey("b"))
        prod = Prod(input=a_ref, result=b_ref)

        assert prod.io.input == a_ref
        assert prod.io.result == b_ref

    def test_wired_producer_both_io_and_kwargs_raises(self):
        """WiredProducer with both io and kwargs raises TypeError."""

        @dataclass
        class IO:
            x: ResourceRef
            y: ResourceRef = out_field()

        class Prod(WiredProducer[IO]):
            def compute(self, reg):
                pass

        registry = Registry()
        registry.declare("a", buffer=np.array([1.0]))
        registry.declare("b")
        a_ref = ResourceRef(registry, ResourceKey("a"))
        b_ref = ResourceRef(registry, ResourceKey("b"))
        io_obj = IO(x=a_ref, y=b_ref)

        with pytest.raises(TypeError, match="either io or kwargs"):
            Prod(io=io_obj, x=a_ref)

    def test_wired_producer_io_not_dataclass_raises(self):
        """WiredProducer with non-dataclass io raises TypeError."""

        class Prod(WiredProducer):
            def compute(self, reg):
                pass

        with pytest.raises(TypeError, match="dataclass IO object"):
            Prod(io={"x": 1})

    def test_wired_producer_output_not_resourceref_raises(self):
        """WiredProducer with non-ResourceRef output field raises."""

        @dataclass
        class IO:
            x: ResourceRef
            y: int = out_field()  # Wrong type for output

        class Prod(WiredProducer[IO]):
            def compute(self, reg):
                pass

        with pytest.raises(TypeError, match="ResourceRef"):
            Prod(io=IO(x=ResourceRef(Registry(), ResourceKey("a")), y=42))

    def test_input_refs_extracts_inputs(self):
        """input_refs returns dict of all input ResourceRefs."""
        registry = Registry()
        registry.declare("a", buffer=np.array([1.0]))
        registry.declare("b", buffer=np.array([2.0]))
        registry.declare("c")

        @dataclass
        class IO:
            in1: ResourceRef
            in2: ResourceRef
            output: ResourceRef = out_field()

        class Prod(WiredProducer[IO]):
            def compute(self, reg):
                pass

        a_ref = ResourceRef(registry, ResourceKey("a"))
        b_ref = ResourceRef(registry, ResourceKey("b"))
        c_ref = ResourceRef(registry, ResourceKey("c"))
        io = IO(in1=a_ref, in2=b_ref, output=c_ref)
        prod = Prod(io=io)

        input_refs = prod.input_refs()
        assert "in1" in input_refs
        assert "in2" in input_refs
        assert "output" not in input_refs
        assert input_refs["in1"] == a_ref
        assert input_refs["in2"] == b_ref

    def test_output_refs_extracts_outputs(self):
        """output_refs returns dict of all output ResourceRefs."""
        registry = Registry()
        registry.declare("a", buffer=np.array([1.0]))
        registry.declare("b")
        registry.declare("c")

        @dataclass
        class IO:
            input: ResourceRef
            out1: ResourceRef = out_field()
            out2: ResourceRef = out_field()

        class Prod(WiredProducer[IO]):
            def compute(self, reg):
                pass

        a_ref = ResourceRef(registry, ResourceKey("a"))
        b_ref = ResourceRef(registry, ResourceKey("b"))
        c_ref = ResourceRef(registry, ResourceKey("c"))
        io = IO(input=a_ref, out1=b_ref, out2=c_ref)
        prod = Prod(io=io)

        output_refs = prod.output_refs()
        assert "out1" in output_refs
        assert "out2" in output_refs
        assert "input" not in output_refs
        assert output_refs["out1"] == b_ref
        assert output_refs["out2"] == c_ref

    def test_ensure_outputs_with_custom_alloc(self):
        """ensure_outputs uses custom alloc function from out_field()."""
        registry = Registry()
        registry.declare("a", buffer=np.array([1.0]))
        registry.declare("b")

        def custom_alloc(reg, io):
            return np.array([99.0])

        @dataclass
        class IO:
            input: ResourceRef
            output: ResourceRef = out_field(alloc=custom_alloc)

        class Prod(WiredProducer[IO]):
            def compute(self, reg):
                pass

        a_ref = ResourceRef(registry, ResourceKey("a"))
        b_ref = ResourceRef(registry, ResourceKey("b"))
        io = IO(input=a_ref, output=b_ref)
        prod = Prod(io=io)

        outputs = prod.ensure_outputs(registry, strict=False, require_shape=False)
        assert "output" in outputs
        # Custom alloc should have been called
        assert np.array_equal(b_ref.peek(), np.array([99.0]))

    def test_ensure_outputs_with_spec_shape(self):
        """ensure_outputs allocates using spec shape."""
        from rheidos.compute import ResourceSpec

        registry = Registry()
        registry.declare("a", buffer=np.array([1.0]))

        spec = ResourceSpec(kind="numpy", dtype=np.float32, shape=(5,))
        key = ResourceKey("b", spec=spec)
        registry.declare("b", spec=spec)

        @dataclass
        class IO:
            input: ResourceRef
            output: ResourceRef = out_field()

        class Prod(WiredProducer[IO]):
            def compute(self, reg):
                pass

        a_ref = ResourceRef(registry, ResourceKey("a"))
        b_ref = ResourceRef(registry, key)
        io = IO(input=a_ref, output=b_ref)
        prod = Prod(io=io)

        outputs = prod.ensure_outputs(registry, require_shape=False, strict=False)
        assert "output" in outputs
        buf = b_ref.peek()
        assert buf is not None
        assert buf.shape == (5,)

    def test_ensure_outputs_missing_shape_strict_raises(self):
        """ensure_outputs raises if output shape is missing in strict mode."""
        registry = Registry()
        registry.declare("a", buffer=np.array([1.0]))
        registry.declare("b")

        @dataclass
        class IO:
            input: ResourceRef
            output: ResourceRef = out_field()

        class Prod(WiredProducer[IO]):
            def compute(self, reg):
                pass

        a_ref = ResourceRef(registry, ResourceKey("a"))
        b_ref = ResourceRef(registry, ResourceKey("b"))  # No spec with shape
        io = IO(input=a_ref, output=b_ref)
        prod = Prod(io=io)

        # strict=True, require_shape=True, no shape -> should raise
        with pytest.raises(RuntimeError, match="shape"):
            prod.ensure_outputs(registry, require_shape=True, strict=True)

    def test_ensure_outputs_realloc_when_mismatched(self):
        """ensure_outputs with realloc=True rereallocates mismatched buffers."""
        from rheidos.compute import ResourceSpec

        registry = Registry()
        registry.declare("a", buffer=np.array([1.0]))

        spec = ResourceSpec(kind="numpy", dtype=np.float32, shape=(5,))
        registry.declare("b", spec=spec)  # No buffer initially

        @dataclass
        class IO:
            input: ResourceRef
            output: ResourceRef = out_field()

        class Prod(WiredProducer[IO]):
            def compute(self, reg):
                pass

        a_ref = ResourceRef(registry, ResourceKey("a"))
        b_ref = ResourceRef(registry, ResourceKey("b", spec=spec))
        io = IO(input=a_ref, output=b_ref)
        prod = Prod(io=io)

        # ensure_outputs with realloc=True allocates when None
        outputs = prod.ensure_outputs(
            registry, realloc=True, require_shape=False, strict=False
        )
        assert "output" in outputs
        buf = b_ref.peek()
        assert buf is not None
        assert buf.shape == (5,)

    def test_ensure_outputs_no_spec_returns_none(self):
        """ensure_outputs returns None buffer if no spec."""
        registry = Registry()
        registry.declare("a", buffer=np.array([1.0]))
        registry.declare("b")

        @dataclass
        class IO:
            input: ResourceRef
            output: ResourceRef = out_field()

        class Prod(WiredProducer[IO]):
            def compute(self, reg):
                pass

        a_ref = ResourceRef(registry, ResourceKey("a"))
        b_ref = ResourceRef(registry, ResourceKey("b"))  # No spec
        io = IO(input=a_ref, output=b_ref)
        prod = Prod(io=io)

        outputs = prod.ensure_outputs(registry, require_shape=False, strict=False)
        assert "output" in outputs
        # Buffer should still be None if no spec
        assert b_ref.peek() is None
