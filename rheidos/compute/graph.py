from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

from .registry import ProducerBase, Registry
from .resource import Resource
from .world import ModuleKey, World


def _fmt_list(items: Sequence[str]) -> str:
    if not items:
        return "[]"
    return "[" + ", ".join(items) + "]"


def _dedupe(items: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _producer_label(producer: ProducerBase) -> str:
    cls = producer.__class__
    return f"{cls.__module__}.{cls.__qualname__}"


def _module_label(key: ModuleKey) -> str:
    scope, module_cls = key
    module_name = getattr(module_cls, "NAME", module_cls.__name__)
    prefix = f"{scope}.{module_name}" if scope else module_name
    if module_name != module_cls.__name__:
        return f"{prefix} ({module_cls.__name__})"
    return prefix


def _resource_list(reg: Registry, *, sort: bool) -> List[Resource]:
    resources = list(reg._res.values())
    if sort:
        resources.sort(key=lambda r: r.name)
    return resources


def _dot_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _dot_node_id(prefix: str, value: str) -> str:
    return f'{prefix}{_dot_escape(value)}'


def _dot_node(
    node_id: str, *, label: str, shape: str, extra_attrs: Sequence[str] | None = None
) -> str:
    attrs = [f'label="{_dot_escape(label)}"', f'shape="{shape}"']
    if extra_attrs:
        attrs.extend(extra_attrs)
    return f'  "{node_id}" [{", ".join(attrs)}];'


def _format_resources(reg: Registry, *, sort: bool) -> List[str]:
    resources = _resource_list(reg, sort=sort)
    if not resources:
        return ["- <none>"]

    lines: List[str] = []
    for res in resources:
        producer = _producer_label(res.producer) if res.producer else "None"
        lines.append(
            f"- {res.name} | producer={producer} | deps={_fmt_list(res.deps)}"
        )
    return lines


def _collect_producer_outputs(resources: Sequence[Resource]) -> Dict[ProducerBase, List[str]]:
    outputs: Dict[ProducerBase, List[str]] = {}
    for res in resources:
        if res.producer is None:
            continue
        outputs.setdefault(res.producer, []).append(res.name)
    return outputs


def _producer_deps(
    resources_by_name: Dict[str, Resource], outputs: Sequence[str]
) -> List[str]:
    deps: List[str] = []
    for out in outputs:
        res = resources_by_name.get(out)
        if res is None:
            continue
        deps.extend(res.deps)
    return _dedupe(deps)


def _format_producers(reg: Registry, *, sort: bool) -> List[str]:
    resources = _resource_list(reg, sort=sort)
    outputs_by_prod = _collect_producer_outputs(resources)
    if not outputs_by_prod:
        return ["- <none>"]

    resources_by_name = {res.name: res for res in resources}

    items = list(outputs_by_prod.items())
    if sort:
        items.sort(key=lambda item: _producer_label(item[0]))

    lines: List[str] = []
    for producer, outputs in items:
        outputs_list = sorted(outputs) if sort else list(outputs)
        deps = _producer_deps(resources_by_name, outputs_list)
        lines.append(
            f"- {_producer_label(producer)} | outputs={_fmt_list(outputs_list)} | deps={_fmt_list(deps)}"
        )
    return lines


def _format_modules(world: World, *, sort: bool) -> List[str]:
    deps = world.module_dependencies()
    if not deps:
        return ["- <none>"]

    keys = list(deps.keys())
    if sort:
        keys.sort(key=_module_label)

    lines: List[str] = []
    for key in keys:
        child_labels = [_module_label(k) for k in deps.get(key, set())]
        if sort:
            child_labels.sort()
        lines.append(f"- {_module_label(key)} | requires={_fmt_list(child_labels)}")
    return lines


def format_dependency_graph_dot(
    world: World,
    *,
    include_resources: bool = True,
    include_producers: bool = True,
    include_modules: bool = False,
    sort: bool = True,
    rankdir: str = "LR",
) -> str:
    reg = world.reg
    resources = _resource_list(reg, sort=sort)
    resources_by_name = {res.name: res for res in resources}
    outputs_by_prod = _collect_producer_outputs(resources)

    lines: List[str] = []
    lines.append("digraph dependency_graph {")
    lines.append(f'  rankdir="{_dot_escape(rankdir)}";')
    lines.append('  node [fontname="Helvetica"];')

    if include_resources:
        for res in resources:
            node_id = _dot_node_id("res::", res.name)
            lines.append(_dot_node(node_id, label=res.name, shape="box"))

    if include_producers:
        producers = list(outputs_by_prod.items())
        if sort:
            producers.sort(key=lambda item: _producer_label(item[0]))
        for producer, _outputs in producers:
            label = _producer_label(producer)
            node_id = _dot_node_id("prod::", label)
            lines.append(
                _dot_node(
                    node_id,
                    label=label,
                    shape="ellipse",
                    extra_attrs=['style="filled"', 'fillcolor="lightgray"'],
                )
            )

        for producer, outputs in producers:
            prod_label = _producer_label(producer)
            prod_id = _dot_node_id("prod::", prod_label)
            deps = _producer_deps(resources_by_name, outputs)
            for dep in deps:
                dep_id = _dot_node_id("res::", dep)
                lines.append(f'  "{dep_id}" -> "{prod_id}";')
            for out in outputs:
                out_id = _dot_node_id("res::", out)
                lines.append(f'  "{prod_id}" -> "{out_id}";')
    else:
        for res in resources:
            res_id = _dot_node_id("res::", res.name)
            for dep in res.deps:
                dep_id = _dot_node_id("res::", dep)
                lines.append(f'  "{dep_id}" -> "{res_id}";')

    if include_modules:
        module_deps = world.module_dependencies()
        if module_deps:
            lines.append('  subgraph cluster_modules {')
            lines.append('    label="Modules";')
            lines.append('    color="lightgray";')
            module_keys = list(module_deps.keys())
            if sort:
                module_keys.sort(key=_module_label)
            for key in module_keys:
                label = _module_label(key)
                node_id = _dot_node_id("mod::", label)
                lines.append(
                    '    '
                    + _dot_node(
                        node_id,
                        label=label,
                        shape="folder",
                        extra_attrs=['style="filled"', 'fillcolor="white"'],
                    )
                )
            for parent, children in module_deps.items():
                parent_id = _dot_node_id("mod::", _module_label(parent))
                child_list = list(children)
                if sort:
                    child_list.sort(key=_module_label)
                for child in child_list:
                    child_id = _dot_node_id("mod::", _module_label(child))
                    lines.append(f'    "{parent_id}" -> "{child_id}";')
            lines.append("  }")

    lines.append("}")
    return "\n".join(lines).rstrip()


def export_dependency_graph_dot(
    world: World,
    path: str,
    *,
    include_resources: bool = True,
    include_producers: bool = True,
    include_modules: bool = False,
    sort: bool = True,
    rankdir: str = "LR",
) -> str:
    dot = format_dependency_graph_dot(
        world,
        include_resources=include_resources,
        include_producers=include_producers,
        include_modules=include_modules,
        sort=sort,
        rankdir=rankdir,
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(dot)
        handle.write("\n")
    return dot


def format_dependency_graph(
    world: World,
    *,
    include_resources: bool = True,
    include_producers: bool = True,
    include_modules: bool = True,
    sort: bool = True,
) -> str:
    lines: List[str] = []

    if include_resources:
        lines.append("Resources")
        lines.extend(_format_resources(world.reg, sort=sort))

    if include_producers:
        if lines:
            lines.append("")
        lines.append("Producers")
        lines.extend(_format_producers(world.reg, sort=sort))

    if include_modules:
        if lines:
            lines.append("")
        lines.append("Modules")
        lines.extend(_format_modules(world, sort=sort))

    return "\n".join(lines).rstrip()


def print_dependency_graph(
    world: World,
    *,
    include_resources: bool = True,
    include_producers: bool = True,
    include_modules: bool = True,
    sort: bool = True,
) -> None:
    print(
        format_dependency_graph(
            world,
            include_resources=include_resources,
            include_producers=include_producers,
            include_modules=include_modules,
            sort=sort,
        )
    )
