import dolfin as dl
import mshr as ms

domain = ms.Rectangle(dl.Point(0.,0.), dl.Point(1.,1.))-ms.Rectangle(dl.Point(0.25,0.15),dl.Point(0.5,0.4))-ms.Rectangle(dl.Point(0.6,0.6),dl.Point(0.75,0.85))
mesh = ms.generate_mesh(domain, 18)
dl.File("test.xml") << mesh
