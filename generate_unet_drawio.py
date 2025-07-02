# Save this script as generate_unet_drawio.py and run it
from xml.etree.ElementTree import Element, SubElement, tostring
import xml.dom.minidom

# Define the blocks
blocks = [
    ("Input", "Input\nShape: (None,None,None,1)"),
    ("Encoder1", "Conv3D(16)+ReLU\nConv3D(16)+ReLU\nMaxPool(2x2x2)"),
    ("Encoder2", "Conv3D(32)+ReLU\nConv3D(32)+ReLU\nMaxPool(2x2x2)"),
    ("Encoder3", "Conv3D(64)+ReLU\nConv3D(64)+ReLU\nMaxPool(2x2x2)"),
    ("Bottleneck", "Conv3D(128)+ReLU\nConv3D(128)+ReLU"),
    ("Decoder1", "UpSampling(2x2x2)\nConcat\nConv3D(64)+ReLU\nConv3D(64)+ReLU"),
    ("Decoder2", "UpSampling(2x2x2)\nConcat\nConv3D(32)+ReLU\nConv3D(32)+ReLU"),
    ("Decoder3", "UpSampling(2x2x2)\nConcat\nConv3D(16)+ReLU\nConv3D(16)+ReLU"),
    ("Output", "Conv3D(1)\nSigmoid"),
]

# XML Root
root = Element('mxfile', host='app.diagrams.net')
diagram = SubElement(root, 'diagram', name="U-Net Fault Segmentation")

# Graph Model
graph = SubElement(diagram, 'mxGraphModel', dx="1000", dy="1000", grid="1", gridSize="10", guides="1")
root_cell = SubElement(graph, 'root')

# Layer
layer = SubElement(root_cell, 'mxCell', id="0")
layer_cell = SubElement(root_cell, 'mxCell', id="1", parent="0")

# Block placement parameters
x_spacing = 220
y_base = 100
block_width = 160
block_height = 60

# Add Blocks
cells = {}
for idx, (id_name, label) in enumerate(blocks):
    cell_id = str(idx + 2)
    x = idx * x_spacing
    y = y_base
    cell = SubElement(root_cell, 'mxCell',
                      id=cell_id,
                      value=label,
                      style="shape=rectangle;rounded=1;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;",
                      vertex="1",
                      parent="1")
    geometry = SubElement(cell, 'mxGeometry',
                          x=str(x),
                          y=str(y),
                          width=str(block_width),
                          height=str(block_height))
    geometry.set('as', 'geometry')
    cells[id_name] = cell_id


# Add Arrows
for i in range(len(blocks) - 1):
    SubElement(root_cell, 'mxCell',
               id=str(100 + i),
               style="edgeStyle=orthogonalEdgeStyle;endArrow=block;html=1;strokeColor=#000000;",
               edge="1",
               parent="1",
               source=cells[blocks[i][0]],
               target=cells[blocks[i+1][0]])

# Add skip connections
skip_connections = [
    ("Encoder3", "Decoder1"),
    ("Encoder2", "Decoder2"),
    ("Encoder1", "Decoder3"),
]

for idx, (src, tgt) in enumerate(skip_connections):
    SubElement(root_cell, 'mxCell',
               id=str(200 + idx),
               style="edgeStyle=orthogonalEdgeStyle;dashed=1;endArrow=open;html=1;strokeColor=#ff0000;",
               edge="1",
               parent="1",
               source=cells[src],
               target=cells[tgt])

# Output pretty XML
xml_str = xml.dom.minidom.parseString(tostring(root)).toprettyxml(indent="  ")

# Write to file
with open("unet_fault_segmentation.drawio", "w") as f:
    f.write(xml_str)

print("✅ Done! File 'unet_fault_segmentation.drawio' created.")
