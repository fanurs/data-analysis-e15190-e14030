import pytest

import itertools
import tempfile

import matplotlib as mpl
mpl.use('Agg') # backend plotting, i.e. to suppress window pops up
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from e15190.neutron_wall import geometry as nwgeom
from e15190.utilities import geometry as geom

@pytest.fixture()
def nw_wall():
    return {AB: nwgeom.Wall(AB) for AB in 'AB'}

class TestWall:
    def test_read_from_inventor_readings(self, nw_wall):
        for AB in ('A', 'B'):
            wall = nw_wall[AB]
            bars = wall.read_from_inventor_readings(wall.path_inventor_readings)
            assert len(bars) == 25
            for bar in bars:
                assert isinstance(bar, nwgeom.Bar)

    def test_save_vertices_to_database(self, nw_wall):
        for AB in ('A', 'B'):
            wall = nw_wall[AB]
            bars = wall.read_from_inventor_readings(wall.path_inventor_readings)
            tmp_path = tempfile.NamedTemporaryFile(suffix='.dat').name
            wall.save_vertices_to_database('B', tmp_path, bars)
            df = pd.read_csv(tmp_path, delim_whitespace=True, comment='#')
            assert tuple(df.columns) == ('nwb-bar', 'dir_x', 'dir_y', 'dir_z', 'x', 'y', 'z')
            assert len(df) == 25 * 8 # n_bars * n_vertices
    
    def test_save_pca_to_database(self, nw_wall):
        for AB in ('A', 'B'):
            wall = nw_wall[AB]
            bars = wall.read_from_inventor_readings(wall.path_inventor_readings)
            tmp_path = tempfile.NamedTemporaryFile(suffix='.dat').name
            wall.save_pca_to_database('B', tmp_path, bars)
            df = pd.read_csv(tmp_path, delim_whitespace=True, comment='#')
            assert tuple(df.columns) == ('nwb-bar', 'vector', 'lab-x', 'lab-y', 'lab-z')
            assert len(df) == 25 * 4 # n_bars * (1 mean + 3 components)

    def test___init__(self):
        pyrex_wall = nwgeom.Wall('B', contain_pyrex=True, refresh_from_inventor_readings=False)
        nopyrex_wall = nwgeom.Wall('B', contain_pyrex=False, refresh_from_inventor_readings=False)

        for pyrex_bar, nopyrex_bar in zip(pyrex_wall.bars.values(), nopyrex_wall.bars.values()):
            assert pyrex_bar.contain_pyrex == True
            assert nopyrex_bar.contain_pyrex == False
            assert pyrex_bar.length > nopyrex_bar.length
    
@pytest.fixture
def nw_bars():
    return dict(
        nwb_pyrex=nwgeom.Wall('B', contain_pyrex=True).bars,
        nwb_nopyrex=nwgeom.Wall('B', contain_pyrex=False).bars,
    )

class TestBar:
    def test___init__(self):
        # a hypothetical bar
        vertices = np.array([
            [5, 0, 0],
            [5, 0, 2],
            [5, 1, 0],
            [5, 1, 2],
            [-5, 0, 0],
            [-5, 0, 2],
            [-5, 1, 0],
            [-5, 1, 2],
        ])
        bar = nwgeom.Bar(vertices)
        expected_pca_components = np.array([
            [-1, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
        ], dtype=float)
        assert np.allclose(bar.pca.components_, expected_pca_components)
    
    def test_length(self, nw_bars):
        for bar in nw_bars['nwb_pyrex'].values():
            assert bar.length == pytest.approx(76 * 2.54 + 2 * bar.pyrex_thickness, abs=0.01)
        for bar in nw_bars['nwb_nopyrex'].values():
            assert bar.length == pytest.approx(76 * 2.54, abs=0.01)
    
    def test_height(self, nw_bars):
        for bar in nw_bars['nwb_pyrex'].values():
            assert bar.height == pytest.approx(3 * 2.54 + 2 * bar.pyrex_thickness, abs=0.01)
        for bar in nw_bars['nwb_nopyrex'].values():
            assert bar.height == pytest.approx(3 * 2.54, abs=0.01)
    
    def test_thickness(self, nw_bars):
        for bar in nw_bars['nwb_pyrex'].values():
            assert bar.thickness == pytest.approx(2.5 * 2.54 + 2 * bar.pyrex_thickness, abs=0.01)
        for bar in nw_bars['nwb_nopyrex'].values():
            assert bar.thickness == pytest.approx(2.5 * 2.54, abs=0.01)

    def test_remove_pyrex(self, nw_bars):
        for bar in nw_bars['nwb_pyrex'].values():
            assert bar.contain_pyrex == True
            bar.remove_pyrex()
            assert bar.contain_pyrex == False
            with pytest.raises(Exception) as err:
                bar.remove_pyrex()

        for bar in nw_bars['nwb_nopyrex'].values():
            assert bar.contain_pyrex == False
            with pytest.raises(Exception) as err:
                bar.remove_pyrex()
    
    def test_add_pyrex(self, nw_bars):
        for bar in nw_bars['nwb_pyrex'].values():
            assert bar.contain_pyrex == True
            with pytest.raises(Exception) as err:
                bar.add_pyrex()
        for bar in nw_bars['nwb_nopyrex'].values():
            assert bar.contain_pyrex == False
            bar.add_pyrex()
            assert bar.contain_pyrex == True
            with pytest.raises(Exception) as err:
                bar.add_pyrex()

    def test_to_local_coordinates(self, nw_bars):
        for bar in nw_bars['nwb_pyrex'].values():
            # test single 3-tuple
            loc_coord = bar.to_local_coordinates([0, 0, 0])
            # bar centers should be close to zero along the 1st principal axis of the bar
            assert loc_coord[0] == pytest.approx(0, abs=10.0)
            # bar centers should be around 445 cm from the target along the beam direction
            assert loc_coord[2] == pytest.approx(445, abs=2.0)

            # test 2d-array with one row of 3-tuple
            assert np.allclose(bar.to_local_coordinates([[0, 0, 0]]), [loc_coord])

            # test an array of 3-tuples
            lab_coords = np.array(list(bar.vertices.values()))
            loc_coords = bar.to_local_coordinates(lab_coords)
            norms = np.linalg.norm(loc_coords, axis=1)
            # bar vertices should all be ~100 cm away from the bar centers origin
            assert np.all(np.isclose(norms, 100.0, atol=10.0))
            break
    
    def test_to_lab_coordinates(self, nw_bars):
        for bar in nw_bars['nwb_pyrex'].values():
            # test single 3-tuple
            lab_coords = bar.to_lab_coordinates([0, 0, 0])
            xz_distance = np.sqrt(lab_coords[0]**2 + lab_coords[2]**2)
            # bar centers should be around 445 cm from the target along the beam direction
            assert xz_distance == pytest.approx(445, abs=2.0)

            # test 2d-array with one row of 3-tuple
            assert np.allclose(bar.to_lab_coordinates([[0, 0, 0]]), [lab_coords])

            # test an array of 3-tuples
            loc_coords = np.array(list(bar.loc_vertices.values()))
            lab_coords = bar.to_lab_coordinates(loc_coords)
            norms = np.linalg.norm(lab_coords, axis=1)
            # bar vertices should all be at least 445 cm away from the target
            assert np.all(norms > 445)
            # all should have similar norms
            assert np.all(np.isclose(norms, norms[0], atol=10.0))
            # but not identical
            assert ~np.all(np.isclose(norms, norms[0], atol=5.0))
    
    def test_consistent_frame_transformation(self, nw_bars):
        rand_pos = np.random.uniform(-100, 100, size=(20, 3))
        for bar in nw_bars['nwb_pyrex'].values():
            loc_coords = bar.to_local_coordinates(rand_pos)
            lab_coords = bar.to_lab_coordinates(loc_coords)
            assert np.all(np.isclose(rand_pos, lab_coords))

            lab_coords = bar.to_lab_coordinates(rand_pos)
            loc_coords = bar.to_local_coordinates(lab_coords)
            assert np.all(np.isclose(rand_pos, loc_coords))
    
    def test_is_inside(self, nw_bars):
        for bar in nw_bars['nwb_pyrex'].values():
            # test single 3-tuple
            # bar center should be inside
            assert bar.is_inside([0, 0, 0], frame='local')
            # point far from bar center should not be inside
            assert ~bar.is_inside([1e2, 1e2, 1e2], frame='local')
            # point at the lab origin should not be inside
            assert ~bar.is_inside([0, 0, 0], frame='lab')
            # bar center in lab frame should be inside
            assert bar.is_inside(bar.pca.mean_, frame='lab')

            # test an array of 3-tuples
            # points near the bar center should be inside
            loc_coords = np.random.uniform(-0.1, 0.1, size=(20, 3))
            assert np.all(bar.is_inside(loc_coords, frame='local'))
            # points far from the bar center should not be inside
            loc_coords = np.random.uniform(1e2, 1e4, size=(20, 3))
            assert np.all(~bar.is_inside(loc_coords, frame='local'))
            # points near the lab origin should not be inside
            lab_coords = np.random.uniform(-0.1, 0.1, size=(20, 3))
            assert np.all(~bar.is_inside(lab_coords, frame='lab'))
            # points near the bar center in lab frame should be inside
            lab_coords = bar.pca.mean_ + np.random.uniform(-0.1, 0.1, size=(20, 3))
            assert np.all(bar.is_inside(lab_coords, frame='lab'))

            # some edge cases: corners, edges and faces
            bar_vertices = np.array(list(bar.vertices.values()))
            vertex_pairs = np.array(list(itertools.combinations(bar_vertices, 2)))
            bar_edges = 0.5 * np.sum(vertex_pairs, axis=1)
            vertex_triplets = np.array(list(itertools.combinations(bar_vertices, 3)))
            bar_faces = np.sum(vertex_triplets, axis=1) / 3

            tol = 1e-3 # 10 micrometer should be included
            assert np.all(bar.is_inside(bar_vertices, frame='lab', tol=tol))
            assert np.all(bar.is_inside(bar_edges, frame='lab', tol=tol))
            assert np.all(bar.is_inside(bar_faces, frame='lab', tol=tol))

            tol = 1e-8 # 0.1 nanometer should not be able to include everything
            coords = np.vstack([bar_vertices, bar_edges, bar_faces])
            assert ~np.all(bar.is_inside(coords, frame='lab', tol=tol))
    
    def test_randomize_from_local_x(self, nw_bars):
        for bar in nw_bars['nwb_pyrex'].values():
            tol = 1e-1

            # testing inside the bar
            loc_x = np.random.uniform(-0.5 * bar.length + tol, 0.5 * bar.length - tol, size=(20))

            lab_coords = bar.randomize_from_local_x(loc_x, return_frame='lab')
            assert np.all(bar.is_inside(lab_coords, frame='lab'))

            loc_coords = bar.randomize_from_local_x(loc_x, return_frame='local')
            assert np.all(bar.is_inside(loc_coords, frame='local'))
            assert np.allclose(loc_x, loc_coords[:, 0])

            # testing outside the bar
            loc_x = np.hstack([
                np.random.uniform(bar.length + tol, 2 * bar.length, size=(10)),
                np.random.uniform(-2 * bar.length, -bar.length - tol, size=(10)),
            ])

            lab_coords = bar.randomize_from_local_x(loc_x, return_frame='lab')
            assert ~np.any(bar.is_inside(lab_coords, frame='lab'))

            loc_coords = bar.randomize_from_local_x(loc_x, return_frame='local')
            assert ~np.any(bar.is_inside(loc_coords, frame='local'))
            assert np.allclose(loc_x, loc_coords[:, 0])

            # testing local_ynorm
            loc_coords = bar.randomize_from_local_x(loc_x, local_ynorm=0, return_frame='local')
            assert np.allclose(loc_coords[:, 1], 0)
            loc_coords = bar.randomize_from_local_x(loc_x, local_ynorm=[-0.2, 0.2], return_frame='local')
            assert np.allclose(loc_coords[:, 1], 0, atol=0.2 * bar.height + 1e-6)
            assert ~np.allclose(loc_coords[:, 1], 0, atol=1e-10)

            # testing local_znorm
            loc_coords = bar.randomize_from_local_x(loc_x, local_znorm=0, return_frame='local')
            assert np.allclose(loc_coords[:, 2], 0)
            loc_coords = bar.randomize_from_local_x(loc_x, local_znorm=[-0.2, 0.2], return_frame='local')
            assert np.allclose(loc_coords[:, 2], 0, atol=0.2 * bar.thickness + 1e-6)
            assert ~np.allclose(loc_coords[:, 2], 0, atol=1e-10)

            # testing random_seed
            rseed = np.random.randint(0, 100)
            # without fixed random seed, the randomization should be different
            coords0 = bar.randomize_from_local_x(loc_x)
            coords1 = bar.randomize_from_local_x(loc_x)
            assert ~np.allclose(coords0, coords1)
            # with fixed random seed, the randomization should be the same
            coords0 = bar.randomize_from_local_x(loc_x, random_seed=rseed)
            coords1 = bar.randomize_from_local_x(loc_x, random_seed=rseed)
            assert np.allclose(coords0, coords1)

    def test_construct_plotly_mesh3d(self, nw_bars):
        for bar in nw_bars['nwb_pyrex'].values():
            bar.construct_plotly_mesh3d()
            bar_vertices = [tuple(np.round(vertex, decimals=2)) for vertex in bar.vertices.values()]

            triangles = bar.triangle_mesh.get_triangles()
            tri_vertices = np.unique(np.round(triangles.reshape(-1, 3), decimals=2), axis=0)
            tri_vertices = [tuple(vertex) for vertex in tri_vertices]
            assert len(tri_vertices) == 8
            assert set(tri_vertices) == set(bar_vertices)
        
    def test_simple_simulation(self, nw_bars):
        n_rays = 20
        for bar in nw_bars['nwb_nopyrex'].values():
            result = bar.simple_simulation(n_rays=n_rays)
            assert result['intersections'].shape == (12, n_rays, 3)
            # rough check on coverage
            assert 20 < np.degrees(result['polar_range'][0]) < 40
            assert 40 < np.degrees(result['polar_range'][1]) < 60
            assert -35 < np.degrees(result['azimuth_range'][0]) < 35
            assert -35 < np.degrees(result['azimuth_range'][1]) < 35

            # every n_rays should have either zero or two intersections
            intersections = np.swapaxes(result['intersections'], 0, 1)
            norms = np.linalg.norm(intersections, axis=2)
            assert set(np.count_nonzero(norms, axis=1)).issubset({0, 2})

            # check random_seed
            rseeds = np.random.randint(0, 100)
            result0 = bar.simple_simulation(n_rays=n_rays, random_seed=rseeds)
            assert ~np.allclose(result['intersections'], result0['intersections'])
            result1 = bar.simple_simulation(n_rays=n_rays, random_seed=rseeds)
            assert np.allclose(result0['intersections'], result1['intersections'])
    
    def test_get_hit_positions(self, nw_bars):
        n_rays = 20
        for bar in nw_bars['nwb_nopyrex'].values():
            bar.simple_simulation(n_rays=n_rays)

            # default should be
            # hit_t = 'uniform
            # frame = 'local
            # coordinate = 'cartesian'
            rseed = np.random.randint(0, 100)
            hits = bar.get_hit_positions(random_seed=rseed)
            hits0 = bar.get_hit_positions(
                hit_t='uniform', frame='local', coordinate='cartesian',
                random_seed=rseed,
            )
            assert np.allclose(hits, hits0)

            # all hits are inside the bar
            n_hits = len(hits)
            if n_hits > 0:
                assert np.all(bar.is_inside(hits, frame='local'))
            
            # spherical lab frame
            hits = bar.get_hit_positions(frame='lab', coordinate='spherical')
            assert len(hits) == n_hits
            if n_hits > 0:
                hits = geom.spherical_to_cartesian(hits)
                assert np.all(bar.is_inside(hits, frame='lab'))
            
            # cartesian lab frame
            hits = bar.get_hit_positions(frame='lab', coordinate='cartesian')
            assert len(hits) == n_hits
            if n_hits > 0:
                assert np.all(bar.is_inside(hits, frame='lab'))

            # testing constant hit_t values
            hits0 = bar.get_hit_positions(hit_t=0)
            hits1 = bar.get_hit_positions(hit_t=1)
            hits_out_pos = bar.get_hit_positions(hit_t=2)
            hits_out_neg = bar.get_hit_positions(hit_t=-1)
            if n_hits > 0:
                assert np.all(bar.is_inside(hits0, frame='local'))
                assert np.all(bar.is_inside(hits1, frame='local'))
                assert np.all(bar.is_inside(0.5 * (hits0 + hits1), frame='local'))
                assert np.all(~bar.is_inside(hits_out_pos, frame='local'))
                assert np.all(~bar.is_inside(hits_out_neg, frame='local'))
            
            # testing a custom callable hit_t
            hit_t = lambda size: np.clip(0, 1, np.random.exponential(0.2, size=size))
            hits = bar.get_hit_positions(hit_t=hit_t)
            n_hits = len(hits)
            if n_hits > 0:
                assert np.all(bar.is_inside(hits, frame='local'))
    
    def test_draw_hit_pattern2d(self, nw_bars):
        fig, ax = plt.subplots()
        for bar in nw_bars['nwb_pyrex'].values():
            bar.simple_simulation(3)

            # spherical lab frame
            hits = bar.get_hit_positions(frame='lab', coordinate='spherical')
            hist = bar.draw_hit_pattern2d(hits, ax=ax, frame='lab', coordinate='spherical')
            # check if the histogram covers the entire bar
            assert len(hits) == pytest.approx(np.sum(hist[0]))

            # cartesian local frame
            hits = bar.get_hit_positions(frame='local', coordinate='cartesian')
            hist = bar.draw_hit_pattern2d(hits, ax=ax, frame='local', coordinate='cartesian')
            # check if the histogram covers the entire bar
            assert len(hits) == pytest.approx(np.sum(hist[0]))
        plt.close()
