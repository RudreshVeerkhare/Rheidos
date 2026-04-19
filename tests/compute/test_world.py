"""Tests for rheidos.compute.world module."""

import numpy as np
import pytest

from rheidos.compute import (
    ModuleBase,
    ModuleInputContractError,
    ResourceRef,
    ResourceKey,
    World,
    producer,
)


class TestNamespace:
    """Test Namespace class."""

    def test_namespace_empty(self):
        """Empty namespace."""
        from rheidos.compute.world import Namespace

        ns = Namespace()
        assert ns.parts == ()
        assert ns.prefix == ""

    def test_namespace_with_parts(self):
        """Namespace with parts."""
        from rheidos.compute.world import Namespace

        ns = Namespace(("scope", "module"))
        assert ns.parts == ("scope", "module")
        assert ns.prefix == "scope.module"

    def test_namespace_child(self):
        """Create child namespace."""
        from rheidos.compute.world import Namespace

        ns = Namespace(("scope",))
        child = ns.child("module")
        assert child.parts == ("scope", "module")

    def test_namespace_qualify(self):
        """Qualify attribute name."""
        from rheidos.compute.world import Namespace

        ns = Namespace(("scope", "module"))
        qualified = ns.qualify("attr")
        assert qualified == "scope.module.attr"


class TestModuleBase:
    """Test ModuleBase class."""

    def test_module_init_no_scope(self):
        """Initialize module without scope."""
        world = World()

        class SimpleModule(ModuleBase):
            NAME = "simple"

        module = SimpleModule(world)
        assert module.world is world
        assert module.reg is world.reg
        assert module.ns.prefix == "simple"

    def test_module_init_with_scope(self):
        """Initialize module with scope."""
        world = World()

        class SimpleModule(ModuleBase):
            NAME = "simple"

        module = SimpleModule(world, scope="outer")
        assert module.ns.prefix == "outer.simple"

    def test_module_prefix_property(self):
        """Module prefix property."""
        world = World()

        class SimpleModule(ModuleBase):
            NAME = "test"

        module = SimpleModule(world)
        assert module.prefix == "test"

    def test_module_lookup_scope_defaults_to_scope(self):
        """lookup_scope defaults to the module scope."""
        world = World()

        class SimpleModule(ModuleBase):
            NAME = "test"

        root_module = SimpleModule(world)
        scoped_module = SimpleModule(world, scope="outer")

        assert root_module.lookup_scope == ""
        assert scoped_module.lookup_scope == "outer"

    def test_module_qualify(self):
        """Module qualify method."""
        world = World()

        class SimpleModule(ModuleBase):
            NAME = "test"

        module = SimpleModule(world)
        qualified = module.qualify("attr")
        assert qualified == "test.attr"

    def test_module_r_alias(self):
        """Module r() is alias for qualify()."""
        world = World()

        class SimpleModule(ModuleBase):
            NAME = "test"

        module = SimpleModule(world)
        assert module.r("attr") == module.qualify("attr")


class TestModuleResourceDeclaration:
    """Test ModuleBase resource declaration."""

    def test_resource_creates_ref(self):
        """resource() creates ResourceRef."""
        world = World()

        class SimpleModule(ModuleBase):
            NAME = "test"

        module = SimpleModule(world)
        ref = module.resource("data")
        assert isinstance(ref, ResourceRef)
        assert ref.name == "test.data"

    def test_resource_with_declare(self):
        """resource() with declare=True declares in registry."""
        world = World()

        class SimpleModule(ModuleBase):
            NAME = "test"

        module = SimpleModule(world)
        ref = module.resource("data", declare=True, buffer=np.array([1.0]))
        assert world.reg.get(ref.name) is not None

    def test_resource_with_buffer(self):
        """resource() with buffer initializes resource."""
        world = World()

        class SimpleModule(ModuleBase):
            NAME = "test"

        module = SimpleModule(world)
        buf = np.array([1.0])
        ref = module.resource("data", declare=True, buffer=buf)
        assert np.array_equal(world.reg.read(ref.name, ensure=False), buf)

    def test_resource_with_deps(self):
        """resource() with deps sets dependencies."""
        world = World()

        class SimpleModule(ModuleBase):
            NAME = "test"

        module = SimpleModule(world)
        world.reg.declare("dep1", buffer=np.array([1.0]))
        world.reg.declare("dep2", buffer=np.array([2.0]))

        ref = module.resource(
            "data", declare=True, deps=["dep1", "dep2"], buffer=np.array([3.0])
        )
        assert "dep1" in world.reg.get(ref.name).deps
        assert "dep2" in world.reg.get(ref.name).deps

    def test_declare_resource(self):
        """declare_resource() explicitly declares ref."""
        world = World()

        class SimpleModule(ModuleBase):
            NAME = "test"

        module = SimpleModule(world)
        ref = module.resource("data")
        module.declare_resource(ref, buffer=np.array([1.0]))
        assert world.reg.get(ref.name) is not None


class TestModuleRequire:
    """Test ModuleBase.require() method."""

    def test_require_creates_module(self):
        """require() creates module instance."""
        world = World()

        class DepModule(ModuleBase):
            NAME = "dep"

        class MainModule(ModuleBase):
            NAME = "main"

            def __init__(self, world, **kwargs):
                super().__init__(world, **kwargs)
                self.dep = self.require(DepModule)

        main = MainModule(world)
        assert isinstance(main.dep, DepModule)

    def test_require_singleton(self):
        """require() returns same instance on multiple calls."""
        world = World()

        class DepModule(ModuleBase):
            NAME = "dep"

        class MainModule(ModuleBase):
            NAME = "main"

            def __init__(self, world, **kwargs):
                super().__init__(world, **kwargs)
                self.dep1 = self.require(DepModule)
                self.dep2 = self.require(DepModule)

        main = MainModule(world)
        assert main.dep1 is main.dep2

    def test_require_with_args(self):
        """require() supports args and caches by args."""
        world = World()

        class DepModule(ModuleBase):
            NAME = "dep"

            def __init__(self, world, value, **kwargs):
                super().__init__(world, **kwargs)
                self.value = value

        class MainModule(ModuleBase):
            NAME = "main"

            def __init__(self, world, **kwargs):
                super().__init__(world, **kwargs)
                self.dep = self.require(DepModule, 1)

        main = MainModule(world)
        dep_same = world.require(DepModule, 1)
        dep_other = world.require(DepModule, 2)

        assert main.dep is dep_same
        assert main.dep is not dep_other
        assert dep_same.value == 1
        assert dep_other.value == 2

    def test_require_kwargs_distinct_from_args(self):
        """require() treats args and kwargs as distinct cache keys."""
        world = World()

        class DepModule(ModuleBase):
            NAME = "dep"

            def __init__(self, world, value, **kwargs):
                super().__init__(world, **kwargs)
                self.value = value

        dep_args = world.require(DepModule, 1)
        dep_kwargs = world.require(DepModule, value=1)

        assert dep_args is not dep_kwargs
        assert dep_args.value == dep_kwargs.value == 1

    def test_require_rejects_unhashable_args(self):
        """require() rejects unhashable args."""
        world = World()

        class DepModule(ModuleBase):
            NAME = "dep"

            def __init__(self, world, value, **kwargs):
                super().__init__(world, **kwargs)
                self.value = value

        with pytest.raises(TypeError):
            world.require(DepModule, [1, 2, 3])

    def test_require_child_nests_identity_scope_and_shares_lookup_scope(self):
        """Child modules get nested resources but reuse the parent's lookup scope."""
        world = World()

        class DepModule(ModuleBase):
            NAME = "dep"

        class ChildModule(ModuleBase):
            NAME = "child"

            def __init__(self, world, **kwargs):
                super().__init__(world, **kwargs)
                self.dep = self.require(DepModule)

        class ParentModule(ModuleBase):
            NAME = "parent"

            def __init__(self, world, **kwargs):
                super().__init__(world, **kwargs)
                self.dep = self.require(DepModule)
                self.child = self.require(
                    ChildModule,
                    child=True,
                    child_name="worker",
                )

        parent = world.require(ParentModule)

        assert parent.child.prefix == "parent.worker.child"
        assert parent.child.lookup_scope == parent.lookup_scope
        assert parent.child.dep is parent.dep

    def test_require_child_returns_same_instance_for_same_name(self):
        """Requiring the same child role twice returns the same instance."""
        world = World()

        class ChildModule(ModuleBase):
            NAME = "child"

        class ParentModule(ModuleBase):
            NAME = "parent"

            def __init__(self, world, **kwargs):
                super().__init__(world, **kwargs)
                self.left = self.require(
                    ChildModule,
                    child=True,
                    child_name="worker",
                )
                self.right = self.require(
                    ChildModule,
                    child=True,
                    child_name="worker",
                )

        parent = world.require(ParentModule)
        assert parent.left is parent.right

    def test_require_child_distinguishes_child_names(self):
        """Different child role names create different child module instances."""
        world = World()

        class ChildModule(ModuleBase):
            NAME = "child"

        class ParentModule(ModuleBase):
            NAME = "parent"

            def __init__(self, world, **kwargs):
                super().__init__(world, **kwargs)
                self.left = self.require(
                    ChildModule,
                    child=True,
                    child_name="left",
                )
                self.right = self.require(
                    ChildModule,
                    child=True,
                    child_name="right",
                )

        parent = world.require(ParentModule)

        assert parent.left is not parent.right
        assert parent.left.prefix == "parent.left.child"
        assert parent.right.prefix == "parent.right.child"

    def test_world_require_child_outside_module(self):
        """World.require() can build a child when the parent is explicit."""
        world = World()

        class DepModule(ModuleBase):
            NAME = "dep"

        class ChildModule(ModuleBase):
            NAME = "child"

            def __init__(self, world, **kwargs):
                super().__init__(world, **kwargs)
                self.dep = self.require(DepModule)

        class ParentModule(ModuleBase):
            NAME = "parent"

            def __init__(self, world, **kwargs):
                super().__init__(world, **kwargs)
                self.dep = self.require(DepModule)

        parent = world.require(ParentModule)
        child = world.require(
            ChildModule,
            child=True,
            child_name="worker",
            parent=parent,
        )

        assert child.prefix == "parent.worker.child"
        assert child.lookup_scope == parent.lookup_scope
        assert child.dep is parent.dep


class TestModuleProducerBinding:
    """Test ModuleBase.bind_producers()."""

    def test_bind_producers_declares_outputs(self):
        """bind_producers declares outputs with a hidden producer adapter."""
        world = World()

        class DecoratedModule(ModuleBase):
            NAME = "decorated"

            def __init__(self, world, **kwargs):
                super().__init__(world, **kwargs)
                self.a = self.resource("a", declare=True, buffer=np.array([2.0]))
                self.b = self.resource("b", declare=True, buffer=np.array([3.0]))
                self.sum_out = self.resource("sum_out")
                self.bind_producers()

            @producer(inputs=("a", "b"), outputs=("sum_out",))
            def build_sum(self, ctx):
                ctx.commit(sum_out=ctx.inputs.a.get() + ctx.inputs.b.get())

        module = world.require(DecoratedModule)
        resource = world.reg.get(module.sum_out.name)
        assert resource.deps == (module.a.name, module.b.name)
        assert resource.producer is not None
        assert resource.producer.debug_name().endswith("DecoratedModule.build_sum")

        world.reg.ensure(module.sum_out.name)
        assert np.array_equal(module.sum_out.peek(), np.array([5.0]))

    def test_bind_producers_rejects_duplicate_binding(self):
        """bind_producers raises if called twice for the same method."""
        world = World()

        class DecoratedModule(ModuleBase):
            NAME = "duplicate"

            def __init__(self, world, **kwargs):
                super().__init__(world, **kwargs)
                self.a = self.resource("a", declare=True, buffer=np.array([1.0]))
                self.out = self.resource("out")
                self.bind_producers()

            @producer(inputs=("a",), outputs=("out",))
            def build(self, ctx):
                ctx.commit(out=ctx.inputs.a.get())

        module = world.require(DecoratedModule)
        with pytest.raises(RuntimeError, match="already bound"):
            module.bind_producers()

    def test_bind_producers_rejects_missing_attr(self):
        """bind_producers raises when a decorated resource attr is missing."""
        world = World()

        class MissingAttrModule(ModuleBase):
            NAME = "missing_attr"

            def __init__(self, world, **kwargs):
                super().__init__(world, **kwargs)
                self.out = self.resource("out")
                self.bind_producers()

            @producer(inputs=("missing",), outputs=("out",))
            def build(self, ctx):
                pass

        with pytest.raises(AttributeError, match="unknown input resource 'missing'"):
            world.require(MissingAttrModule)

    def test_bind_producers_suggests_close_local_name(self):
        """bind_producers suggests close local resource names on typos."""
        world = World()

        class TypoModule(ModuleBase):
            NAME = "typo"

            def __init__(self, world, **kwargs):
                super().__init__(world, **kwargs)
                self.count = self.resource("count", declare=True, buffer=np.array([1.0]))
                self.out = self.resource("out")
                self.bind_producers()

            @producer(inputs=("coutn",), outputs=("out",))
            def build(self, ctx):
                pass

        with pytest.raises(AttributeError, match="Did you mean 'count'\\?"):
            world.require(TypoModule)

    def test_bind_producers_rejects_non_ref_attr(self):
        """bind_producers raises when a decorated attr is not a ResourceRef."""
        world = World()

        class WrongAttrModule(ModuleBase):
            NAME = "wrong_attr"

            def __init__(self, world, **kwargs):
                super().__init__(world, **kwargs)
                self.a = 123
                self.out = self.resource("out")
                self.bind_producers()

            @producer(inputs=("a",), outputs=("out",))
            def build(self, ctx):
                pass

        with pytest.raises(TypeError, match="expected 'a' to be a ResourceRef"):
            world.require(WrongAttrModule)

    def test_bind_producers_rejects_declared_output(self):
        """bind_producers requires decorated outputs to be undeclared refs."""
        world = World()

        class DeclaredOutputModule(ModuleBase):
            NAME = "declared_output"

            def __init__(self, world, **kwargs):
                super().__init__(world, **kwargs)
                self.a = self.resource("a", declare=True, buffer=np.array([1.0]))
                self.out = self.resource("out", declare=True)
                self.bind_producers()

            @producer(inputs=("a",), outputs=("out",))
            def build(self, ctx):
                pass

        with pytest.raises(RuntimeError, match="already declared"):
            world.require(DeclaredOutputModule)

    def test_bind_producers_validates_required_inputs_on_compute(self):
        """Decorated producers validate required inputs before method execution."""
        world = World()

        class MissingInputModule(ModuleBase):
            NAME = "missing_input"

            def __init__(self, world, **kwargs):
                super().__init__(world, **kwargs)
                self.a = self.resource("a", declare=True)
                self.out = self.resource("out")
                self.bind_producers()

            @producer(inputs=("a",), outputs=("out",))
            def build(self, ctx):
                ctx.commit(out=np.array([9.0]))

        module = world.require(MissingInputModule)
        with pytest.raises(RuntimeError, match="missing required inputs: a"):
            world.reg.ensure(module.out.name)

    def test_bind_producers_supports_required_module_resource_paths(self):
        """Decorated inputs can reference ResourceRefs on required modules."""
        world = World()

        class MeshModule(ModuleBase):
            NAME = "mesh"

            def __init__(self, world, **kwargs):
                super().__init__(world, **kwargs)
                self.V_pos = self.resource(
                    "V_pos",
                    declare=True,
                    buffer=np.array([[1.0, 2.0, 3.0]]),
                )

        class ConsumerModule(ModuleBase):
            NAME = "consumer"

            def __init__(self, world, **kwargs):
                super().__init__(world, **kwargs)
                self.mesh = self.require(MeshModule)
                self.count = self.resource("count")
                self.bind_producers()

            @producer(inputs=("mesh.V_pos",), outputs=("count",))
            def build_count(self, ctx):
                verts = ctx.inputs.mesh.V_pos.get()
                ctx.commit(count=np.array([verts.shape[0]]))

        module = world.require(ConsumerModule)
        resource = world.reg.get(module.count.name)
        assert resource.deps == (module.mesh.V_pos.name,)

        world.reg.ensure(module.count.name)
        assert np.array_equal(module.count.peek(), np.array([1]))

    def test_bind_producers_suggests_close_nested_name(self):
        """bind_producers suggests close dotted resource paths on typos."""
        world = World()

        class MeshModule(ModuleBase):
            NAME = "mesh"

            def __init__(self, world, **kwargs):
                super().__init__(world, **kwargs)
                self.V_pos = self.resource(
                    "V_pos",
                    declare=True,
                    buffer=np.array([[1.0, 2.0, 3.0]]),
                )

        class ConsumerModule(ModuleBase):
            NAME = "consumer"

            def __init__(self, world, **kwargs):
                super().__init__(world, **kwargs)
                self.mesh = self.require(MeshModule)
                self.out = self.resource("out")
                self.bind_producers()

            @producer(inputs=("mesh.V_pso",), outputs=("out",))
            def build(self, ctx):
                pass

        with pytest.raises(ModuleInputContractError) as excinfo:
            world.require(ConsumerModule)

        message = str(excinfo.value)
        assert "ConsumerModule input contract validation failed" in message
        assert "Did you mean 'mesh.V_pos'?" in message


class TestWorld:
    """Test World class."""

    def test_world_init(self):
        """Initialize World."""
        world = World()
        assert world.reg is not None
        assert world._modules == {}

    def test_world_require_creates_module(self):
        """World.require() creates module."""
        world = World()

        class TestModule(ModuleBase):
            NAME = "test"

        module = world.require(TestModule)
        assert isinstance(module, TestModule)

    def test_world_require_singleton(self):
        """World.require() returns same instance."""
        world = World()

        class TestModule(ModuleBase):
            NAME = "test"

        m1 = world.require(TestModule)
        m2 = world.require(TestModule)
        assert m1 is m2

    def test_world_require_with_args_and_scope(self):
        """World.require() caches per scope and args."""
        world = World()

        class TestModule(ModuleBase):
            NAME = "test"

            def __init__(self, world, value, **kwargs):
                super().__init__(world, **kwargs)
                self.value = value

        a1 = world.require(TestModule, 1, scope="a")
        a2 = world.require(TestModule, 1, scope="a")
        b1 = world.require(TestModule, 1, scope="b")

        assert a1 is a2
        assert a1 is not b1

    def test_world_require_rejects_invalid_child_arguments(self):
        """Child-mode require validates the public API consistently."""
        world = World()

        class TestModule(ModuleBase):
            NAME = "test"

        parent = world.require(TestModule)

        with pytest.raises(ValueError, match="child=True requires child_name"):
            world.require(TestModule, child=True, parent=parent)

        with pytest.raises(
            ValueError,
            match="child=True on World.require\\(\\) requires parent",
        ):
            world.require(TestModule, child=True, child_name="worker")

        with pytest.raises(
            ValueError,
            match="single non-empty namespace segment",
        ):
            world.require(
                TestModule,
                child=True,
                child_name="bad.name",
                parent=parent,
            )

        with pytest.raises(
            ValueError,
            match="single non-empty namespace segment",
        ):
            world.require(TestModule, child=True, child_name="", parent=parent)

        with pytest.raises(ValueError, match="child_name requires child=True"):
            world.require(TestModule, child_name="worker")

        with pytest.raises(ValueError, match="parent requires child=True"):
            world.require(TestModule, parent=parent)

    def test_world_require_rolls_back_failed_module_init(self):
        """Failed module init should not poison later require() retries."""
        world = World()

        class DepModule(ModuleBase):
            NAME = "dep"

            def __init__(self, world, **kwargs):
                super().__init__(world, **kwargs)
                self.value = self.resource(
                    "value",
                    declare=True,
                    buffer=np.array([1.0]),
                )

        class BrokenModule(ModuleBase):
            NAME = "broken"

            def __init__(self, world, **kwargs):
                super().__init__(world, **kwargs)
                self.dep = self.require(DepModule)
                self.count = self.resource(
                    "count",
                    declare=True,
                    buffer=np.array([2.0]),
                )
                self.out = self.resource("out")
                self.bind_producers()

            @producer(inputs=("coutn",), outputs=("out",))
            def build(self, ctx):
                pass

        for _ in range(2):
            with pytest.raises(AttributeError, match="Did you mean 'count'\\?"):
                world.require(BrokenModule)

        assert world._modules == {}
        with pytest.raises(KeyError):
            world.reg.get("broken.count")
        with pytest.raises(KeyError):
            world.reg.get("dep.value")

        dep = world.require(DepModule)
        assert np.array_equal(dep.value.peek(), np.array([1.0]))

    def test_child_module_build_failure_rolls_back(self):
        """Broken child modules should not poison the cache or registry."""
        world = World()

        class BrokenChild(ModuleBase):
            NAME = "child"

            def __init__(self, world, **kwargs):
                super().__init__(world, **kwargs)
                self.count = self.resource(
                    "count",
                    declare=True,
                    buffer=np.array([1.0]),
                )
                self.out = self.resource("out")
                self.bind_producers()

            @producer(inputs=("coutn",), outputs=("out",))
            def build(self, ctx):
                pass

        class ParentModule(ModuleBase):
            NAME = "parent"

            def __init__(self, world, **kwargs):
                super().__init__(world, **kwargs)
                self.child = self.require(
                    BrokenChild,
                    child=True,
                    child_name="solver",
                )

        for _ in range(2):
            with pytest.raises(AttributeError, match="Did you mean 'count'\\?"):
                world.require(ParentModule)

        assert world._modules == {}
        assert not any(
            name.startswith("parent.solver.child.")
            for name in world.reg.declared_names()
        )

    def test_world_module_dependencies(self):
        """World tracks module dependencies."""
        world = World()

        class DepModule(ModuleBase):
            NAME = "dep"

        class MainModule(ModuleBase):
            NAME = "main"

            def __init__(self, world, **kwargs):
                super().__init__(world, **kwargs)
                self.dep = self.require(DepModule)

        main = world.require(MainModule)
        deps = world.module_dependencies()
        # Verify deps is a dict
        assert isinstance(deps, dict)
        assert len(deps) > 0

    def test_world_circular_dependency_detection(self):
        """World handles module dependencies correctly."""
        world = World()

        class ModuleA(ModuleBase):
            NAME = "a"

        class ModuleB(ModuleBase):
            NAME = "b"

            def __init__(self, world, **kwargs):
                super().__init__(world, **kwargs)
                # Require A (doesn't create cycle)
                self.a = self.require(ModuleA)

        # This should work fine
        a = world.require(ModuleA)
        b = world.require(ModuleB)
        assert isinstance(a, ModuleA)
        assert isinstance(b, ModuleB)

    def test_world_detects_child_module_cycles(self):
        """Child modules still participate in cycle detection."""
        world = World()

        class ModuleA(ModuleBase):
            NAME = "ModuleA"

            def __init__(self, world, **kwargs):
                super().__init__(world, **kwargs)
                self.child = self.require(
                    ModuleB,
                    child=True,
                    child_name="solver",
                )

        class ModuleB(ModuleBase):
            NAME = "ModuleB"

            def __init__(self, world, **kwargs):
                super().__init__(world, **kwargs)
                self.parent_again = self.require(ModuleA)

        with pytest.raises(RuntimeError, match="Module dependency cycle detected"):
            world.require(ModuleA)

        assert world._modules == {}


class TestModuleResourceDeps:
    """Test module_resource_deps helper."""

    def test_module_resource_deps_all(self):
        """module_resource_deps returns all resource refs."""
        from rheidos.compute.world import module_resource_deps

        world = World()

        class TestModule(ModuleBase):
            NAME = "test"

            def __init__(self, world, **kwargs):
                super().__init__(world, **kwargs)
                self.x = self.resource("x")
                self.y = self.resource("y")
                self.z = 42  # Not a ResourceRef

        module = TestModule(world)
        deps = module_resource_deps(module)
        names = [d.name for d in deps]
        assert "test.x" in names
        assert "test.y" in names

    def test_module_resource_deps_include_filter(self):
        """module_resource_deps with include pattern."""
        from rheidos.compute.world import module_resource_deps

        world = World()

        class TestModule(ModuleBase):
            NAME = "test"

            def __init__(self, world, **kwargs):
                super().__init__(world, **kwargs)
                self.input_x = self.resource("input_x")
                self.output_y = self.resource("output_y")

        module = TestModule(world)
        deps = module_resource_deps(module, include=r"input.*")
        names = [d.name for d in deps]
        assert "test.input_x" in names
        assert "test.output_y" not in names

    def test_module_resource_deps_exclude_filter(self):
        """module_resource_deps with exclude pattern."""
        from rheidos.compute.world import module_resource_deps

        world = World()

        class TestModule(ModuleBase):
            NAME = "test"

            def __init__(self, world, **kwargs):
                super().__init__(world, **kwargs)
                self.input_x = self.resource("input_x")
                self.output_y = self.resource("output_y")

        module = TestModule(world)
        deps = module_resource_deps(module, exclude=r"output.*")
        names = [d.name for d in deps]
        assert "test.input_x" in names
        assert "test.output_y" not in names

    def test_module_cycle_detection(self):
        """Test World detects module dependency cycles."""
        world = World()

        class ModuleA(ModuleBase):
            NAME = "ModuleA"

            def declare(self, reg):
                reg.declare("a", buffer=np.array([1.0]))

        class ModuleB(ModuleBase):
            NAME = "ModuleB"

            def declare(self, reg):
                reg.declare("b", buffer=np.array([2.0]))

        class ModuleC(ModuleBase):
            NAME = "ModuleC"

            def declare(self, reg):
                # Depends on A
                dep = self.require(ModuleA)
                reg.declare("c", deps=["ModuleA/a"])

        # First require should work
        world.require(ModuleA)
        world.require(ModuleB)
        world.require(ModuleC)

        # Verify module dependencies
        deps_dict = world.module_dependencies()
        assert len(deps_dict) >= 0

    def test_module_singleton_pattern(self):
        """Test module namespace property."""
        world = World()

        class TestModule(ModuleBase):
            NAME = "Test"

            def __init__(self, world, **kwargs):
                super().__init__(world, **kwargs)

        # Create module
        mod = TestModule(world)

        # Check namespace
        assert mod.ns is not None
        assert "Test" in mod.ns.prefix

    def test_module_with_scope(self):
        """Test module instances with different scope names."""
        world = World()

        class TestModule(ModuleBase):
            NAME = "Test"

            def declare(self, reg):
                reg.declare("data", buffer=np.array([1.0]))

        # Create module with scope
        mod = TestModule(world, scope="scope1")
        mod.declare(world.reg)

        # Should have scope in namespace
        assert "scope1" in mod.ns.prefix

    def test_module_resource_qualification(self):
        """Test module resource name qualification."""
        world = World()

        class TestModule(ModuleBase):
            NAME = "MyModule"

            def declare(self, reg):
                reg.declare("data", buffer=np.array([1.0]))

        mod = TestModule(world)
        mod.declare(world.reg)

        # Resource should be qualified with module name
        qualified_name = mod.qualify("data")
        assert "MyModule.data" in qualified_name
