#!/usr/bin/env python

import argparse
import meshio
import os
import numpy as np
from configparser import ConfigParser
try:
    from dolfin import XDMFFile, Mesh, MeshValueCollection
    from dolfin.cpp.mesh import MeshFunctionSizet
except ImportError:
    print("Could not import dolfin. Continuing without Dolfin support.")


def msh2xdmf(mesh_name, tdim, gdim, directory="."):
    """
    Function converting a MSH mesh into XDMF files.
    The XDMF files are:
        - "domain.xdmf": the domain;
        - "boundaries.xdmf": the boundaries physical groups from GMSH;
    """

    # Get the mesh name has prefix
    prefix = mesh_name.split('.')[0]
    # Read the input mesh
    msh = meshio.read("{}/{}".format(directory, mesh_name))
    # Generate the domain XDMF file
    export_domain(msh, tdim, gdim, directory, prefix, "domain.xdmf")
    # Generate the boundaries XDMF file
    export_domain(msh, tdim-1, gdim, directory, prefix, "boundaries.xdmf")
    # Export association table
    export_association_table(msh, prefix, directory)


def export_domain(msh, tdim, gdim, directory, prefix, outfix):
    """
    Export the domain XDMF file as well as the subdomains values.
    """
    # Set cell type
    cell_type = ["vertex", "line", "triangle", "tetra"][tdim]
    # Generate the cell block for the domain cells
    data_array = [cell.data for cell in msh.cells if cell.type == cell_type]
    if len(data_array) == 0:
        print("WARNING: No cells found for tdim = {}.".format(tdim))
        data = np.zeros((0, tdim+1), dtype=np.int64)
    else:
        data = np.concatenate(data_array)
    cells = [
        meshio.CellBlock(
            cell_type=cell_type,
            data=data,
        )
    ]
    # Generate the domain cells data (for the subdomains)
    has_physical_groups = False
    if "gmsh:physical" in msh.cell_data:
        cell_data_array = [msh.cell_data["gmsh:physical"][i] \
            for i, cellBlock in enumerate(msh.cells) if cellBlock.type == cell_type]
        if len(cell_data_array) > 0:
            cell_data = { "physical_group_data": [np.concatenate(cell_data_array)] }
            has_physical_groups = True
    if not has_physical_groups:
        print('WARNING: No physical groups found for tdim = {}.'.format(tdim))
        cell_data = {"physical_group_data": [np.zeros((0,1), dtype=np.int64)]}

    # Generate a meshio Mesh for the domain
    domain = meshio.Mesh(
        points=msh.points[:,:gdim],
        cells=cells,
        cell_data=cell_data,
    )
    # Export the XDMF mesh of the domain
    meshio.write(
        "{}/{}_{}".format(directory, prefix, outfix),
        domain,
        file_format="xdmf"
    )

def export_association_table(msh, prefix='mesh', directory='.', verbose=True):
    """
    Display the association between the physical group label and the mesh
    value.
    """
    # Create association table
    association_table = {}

    # Display the correspondance
    formatter = "|{:^20}|{:^20}|"
    topbot = "+{:-^41}+".format("")
    separator = "+{:-^20}+{:-^20}+".format("", "")

    # Display
    if verbose:
        print('\n' + topbot)
        print(formatter.format("GMSH label", "MeshFunction value"))
        print(separator)

    for label, arrays in msh.cell_sets.items():
        # Added check to make sure that the association table
        # doesn't try to import irrelevant information.
        if label != "gmsh:bounding_entities":
            # Get the index of the array in arrays
            for i, array in enumerate(arrays):
                if array.size != 0:
                    index = i
            value = msh.cell_data["gmsh:physical"][index][0]
            # Store the association table in a dictionnary
            association_table[label] = value
            # Display the association
            if verbose:
                print(formatter.format(label, value))
    if verbose:
        print(topbot)
    # Export the association table
    file_content = ConfigParser()
    file_content["ASSOCIATION TABLE"] = association_table
    file_name = "{}/{}_{}".format(directory, prefix, "association_table.ini")
    with open(file_name, 'w') as f:
        file_content.write(f)


def import_mesh(
        prefix="mesh",
        subdomains=False,
        tdim=2, 
        gdim = 2, 
        directory=".",
):
    """Function importing a dolfin mesh.

    Arguments:
        prefix (str, optional): mesh files prefix (eg. my_mesh.msh,
            my_mesh_domain.xdmf, my_mesh_bondaries.xdmf). Defaults to "mesh".
        subdomains (bool, optional): True if there are subdomains. Defaults to
            False.
        dim (int, optional): dimension of the domain. Defaults to 2.
        directory (str, optional): directory of the mesh files. Defaults to ".".

    Output:
        - dolfin Mesh object containing the domain;
        - dolfin MeshFunction object containing the physical lines (dim=2) or
            surfaces (dim=3) defined in the msh file and the sub-domains;
        - association table
    """
    # Set the file name
    domain = "{}_domain.xdmf".format(prefix)
    boundaries = "{}_boundaries.xdmf".format(prefix)

    # create 2 xdmf files if not converted before
    if not os.path.exists("{}/{}".format(directory, domain)) or \
       not os.path.exists("{}/{}".format(directory, boundaries)):
        msh2xdmf("{}.msh".format(prefix), tdim, gdim, directory=directory)

    # Import the converted domain
    mesh = Mesh()
    with XDMFFile("{}/{}".format(directory, domain)) as infile:
        infile.read(mesh)
    # Import the boundaries
    boundaries_mvc = MeshValueCollection("size_t", mesh, dim=tdim) # which dimension?
    with XDMFFile("{}/{}".format(directory, boundaries)) as infile:
        infile.read(boundaries_mvc, 'physical_group_data')
    boundaries_mf = MeshFunctionSizet(mesh, boundaries_mvc)
    # Import the subdomains
    if subdomains:
        subdomains_mvc = MeshValueCollection("size_t", mesh, dim=tdim) # which dimension?
        with XDMFFile("{}/{}".format(directory, domain)) as infile:
            infile.read(subdomains_mvc, 'physical_group_data')
        subdomains_mf = MeshFunctionSizet(mesh, subdomains_mvc)
    # Import the association table
    association_table_name = "{}/{}_{}".format(
        directory, prefix, "association_table.ini")
    file_content = ConfigParser()
    file_content.read(association_table_name)
    association_table = dict(file_content["ASSOCIATION TABLE"])
    # Convert the value from string to int
    for key, value in association_table.items():
        association_table[key] = int(value)
    # Return the Mesh and the MeshFunction objects
    if not subdomains:
        return mesh, boundaries_mf, association_table
    else:
        return mesh, boundaries_mf, subdomains_mf, association_table
