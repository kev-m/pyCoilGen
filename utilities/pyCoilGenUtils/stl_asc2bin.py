"""
This script converts an ASCII format STL file to Binary format.

An ASCII STL file begins with the line:
    solid name

where name is an optional string. The remainder of the line is ignored and may store metadata.

The file continues with any number of triangles, each represented as follows:

    facet normal ni nj nk
        outer loop
            vertex v1x v1y v1z
            vertex v2x v2y v2z
            vertex v3x v3y v3z
        endloop
    endfacet

Whitespace (spaces, tabs, newlines) may be used anywhere in the file except within numbers or words.
The spaces between facet and normal and between outer and loop are required.

The script outputs a binary format STL file.
"""
import argparse
import struct


def read_ascii_stl(file_path):
    """
    Reads an ASCII STL file and extracts the normals and vertices.

    Args:
        file_path (str): Path to the ASCII STL file.

    Returns:
        tuple: Tuple containing a list of normals and a list of vertices.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
        offset = 1
        num_entries = (len(lines) - 2) / 7
        assert num_entries == int(num_entries)
        num_entries = int(num_entries)
        vertices = [None] * num_entries
        normals = [None] * num_entries
        for i in range(num_entries):
            normal = list(map(float, lines[offset + i * 7 + 0].split()[2:]))
            normals[i] = normal
            vertex1 = list(map(float, lines[offset + i * 7 + 2].split()[1:]))
            vertex2 = list(map(float, lines[offset + i * 7 + 3].split()[1:]))
            vertex3 = list(map(float, lines[offset + i * 7 + 4].split()[1:]))
            vertices[i] = [vertex1, vertex2, vertex3]
    return normals, vertices


def write_binary_stl(file_path, normals, vertices):
    with open(file_path, 'wb') as f:
        f.write(b'\x00' * 80)  # Write 80 bytes header
        f.write(struct.pack('<I', len(vertices)))  # Write the number of triangles
        for i in range(len(vertices)):                  # For each triangle...
            f.write(struct.pack('<3f', *normals[i]))        # Write the normal
            for vertex in vertices[i]:
                f.write(struct.pack('<3f', *vertex))        # Write the vertex
            f.write(b'\x00\x00')                            # Write two bytes for attribute byte count (set to 0)


def convert_ascii_to_binary(input_file, output_file):
    """
    Converts an ASCII STL file to binary format.

    Args:
        input_file (str): Path to the input ASCII STL file.
        output_file (str): Path for the output binary STL file.
    """
    normals, vertices = read_ascii_stl(input_file)
    write_binary_stl(output_file, normals, vertices)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert ASCII STL to Binary STL')
    parser.add_argument('input_file', type=str, help='Path to the input ASCII STL file')
    parser.add_argument('output_file', type=str, help='Path for the output binary STL file')

    args = parser.parse_args()
    convert_ascii_to_binary(args.input_file, args.output_file)

    # Example usage
    # convert_ascii_to_binary('../pyCoilGen_Testing/asc_Preoptimzed_SVD_Coil_swept_part1_none.stl',
    #                    'images/bin_Preoptimzed_SVD_Coil_swept_part1_none.stl')
