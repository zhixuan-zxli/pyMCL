import dolfin
from msh2xdmf import import_mesh
from matplotlib import pyplot

mesh, boundaries_mf, subdomains_mf, association_table = import_mesh(prefix='two-phase', subdomains=True, dim=2)

dx = dolfin.Measure('dx', domain=mesh, subdomain_data=subdomains_mf)
ds = dolfin.Measure('ds', domain=mesh, subdomain_data=boundaries_mf)
dS = dolfin.Measure('dS', domain=mesh, subdomain_data=boundaries_mf)

print('Interface measure = {}'.format(dolfin.assemble(1 * dS(association_table["interface"]))))

