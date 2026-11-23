# Lunar base source assets

Third-party source models, terrain data and materials for the redesigned lunar base scene.

## Layout

```text
assets/lunar_base/
├── manifest.yaml      # every asset: shortlist ID, category, source page, author, license, how to fetch
├── fetch_assets.py    # downloads manifest entries into raw/ (idempotent; --only <ID...>)
├── ATTRIBUTION.md     # generated from manifest.yaml
└── raw/               # git-ignored original downloads
    ├── habitat/       # habitat modules, dome, docking hatch
    ├── power/         # distribution cabinets, switches, ORU/battery, PV + masts, cabling
    ├── comms/         # dishes, masts, radome, comm racks, plugs/sockets
    ├── landers/       # detailed LM, Starship backdrop, IM-2 Athena (SRB landers stay in srb_assets)
    ├── logistics/     # carrier, cases (articulated), crates, racks, locker, trolley, stairs
    ├── tools/         # graspable hand/power tools
    ├── isru/          # RASSOR, tanks, valves (jointed), pipes, reels, light tower
    ├── dock/          # rover charger candidates
    ├── vehicles/      # crew rover, ExoMars, LRV scan, NASA SEV
    ├── terrain/       # LOLA Site01 Connecting Ridge DEM, OmniLRS regolith/rocks/crater profiles
    └── materials/     # ambientCG PBR materials (MLI foil, solar cell, white paint, plates)
```

`raw/` holds originals exactly as published (GLB/glTF/SDF/USD/GeoTIFF) and is not committed:
it is several GB and fully reproducible with `python3 fetch_assets.py`.

## Processed scene assets

```text
processing.yaml        # per output: source, exclude/crop, scale or target size, face budget, physics
process/process.py     # one Blender run per output -> usd/<category>/<name>/<name>.usdc (+ textures/, .json)
process/inspect_assets.py, overview_parts.py, label_parts.py   # structure reports + coordinate grids for choosing crops
usd/                   # committed through git LFS (see .gitattributes)
previews/              # git-ignored 4-view renders beside a LunarBot-sized proxy
```

Every output is Z-up, metres, footprint centred at the origin with its lowest point at z = 0.
Static assets carry `UsdPhysics.CollisionAPI` + convex-hull `MeshCollisionAPI` on each mesh (Isaac Lab's
spawner only modifies existing collision APIs, it never adds them); graspables also get a
`RigidBodyAPI` + `MassAPI` on the default prim. Unlit glTF materials are rebuilt as Principled BSDF
before export because Blender's USD exporter drops textures from any other shader graph, and meshes are
welded before decimation because collapse-decimating unwelded triangle soups throws spikes.

## Licenses

- Commit-safe as-is: NASA (public domain — strip NASA insignia from textures used in rendered or
  AI-consumed imagery), CC0, Apache-2.0, BSD-3.
- CC-BY-4.0: commit with the ATTRIBUTION.md entry.
- CC-BY-SA-4.0 (`Z2-04`, `V-03`): derivatives must be released CC-BY-SA; keep them identifiable.
- Excluded by policy: TurboSquid/CGTrader (ML clause), non-commercial or no-derivative licenses,
  NVIDIA `/Isaac/Environments/*`.

## Access notes

- Sketchfab models not in the Objaverse mirror need `SKETCHFAB_API_TOKEN`.
- `Z2-15` (iROSA, Printables) needs a manual download with a Printables login.
