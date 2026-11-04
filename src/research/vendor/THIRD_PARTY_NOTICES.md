# Third-party code vendored in this directory

## echarts.min.js

- Component: Apache ECharts (minified browser bundle), version 6.x
- Source: `https://cdn.jsdelivr.net/npm/echarts@6/dist/echarts.min.js`
- Retrieved: 2026-11-03
- SHA-256: `b66b25aeb4df84e33199dc21694014d336d222cbd9deb0e5a7c14bd6aa0d0fd0`
- License: Apache License 2.0 (the bundle carries its own licence header)

**Why it is vendored rather than linked.** The dashboard must work on a loopback
port with no network, and it must render the same way a reader of the upstream
`lieflat-charts` gallery expects. That skill's Glance family (G18 Draw-in +
Counter, G17 Dynamic Stream, G15/G19 …) is implemented on ECharts; linking a CDN
would make every page load depend on the internet and leak a request from a
monitoring page. The bundle is inlined into the served page at render time, so the
page remains a single self-contained document.

Chart structure, options and animation follow the `lieflat-charts` gallery cards
(see `skills/autoresearch/SKILL.md` § Upstream visual skill); only the data, the
titles and the annotations are ours.
