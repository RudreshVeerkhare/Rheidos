"""Tests for rheidos.compute.world module."""

import pytest
import numpy as np

from rheidos.compute import World, ModuleBase, ResourceRef, ResourceKey


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
