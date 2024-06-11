"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


@partial(jax.jit, static_argnums=(1,))
def apsp(A, n=None):
	"""
	Compute APSP for adjacency matrix A 
	using Seidel's algorithm.
	"""
	if n is None:
		n = A.shape[0]
	assert(n == A.shape[0]), 'n must equal dim of A.'

	n_steps = int(np.ceil(np.log(n)/np.log(2)))
	A_cache = jnp.zeros((n_steps, n, n), dtype=jnp.uint32)
	steps_to_reduce = jnp.array(1, dtype=jnp.int32)

	def _scan_fwd_step(carry, step): 
		i = step
		A, A_cache, steps_to_reduce = carry
		A_cache = A_cache.at[i].set(A)

		Z = A@A
		B = jnp.logical_or(
				A == 1, 
				Z > 0
			).astype(jnp.uint32) \
			 .at[jnp.diag_indices(n)].set(0)
		A = B

		complete = B.sum() - jnp.diagonal(B).sum() == n*(n-1)
		steps_to_reduce += ~complete

		return (A, A_cache, steps_to_reduce), None

	(B, A_cache, steps_to_reduce), _ = jax.lax.scan(
		_scan_fwd_step,
		(A, A_cache, 1),
		jnp.arange(n_steps),
		length=n_steps
	)

	D = 2*B - A_cache[steps_to_reduce-1]

	def _scan_bkwd_step(carry, step):
		i = step
		(T, A_cache,steps_to_reduce) = carry

		A = A_cache[steps_to_reduce - i - 1]
		X = T@A

		thresh = T*(jnp.tile(A.sum(0, keepdims=True), (n, 1)))
		D = 2*T*(X >= thresh) + (2*T - 1)*(X < thresh)
		T = D*(i < steps_to_reduce) + T*(i >= steps_to_reduce)

		return (T, A_cache, steps_to_reduce), None

	(D, _, _), _ = jax.lax.scan(
		_scan_bkwd_step,
		(D, A_cache, steps_to_reduce),
		jnp.arange(1, n_steps),
		length=n_steps-1
	)

	return D


@jax.jit
def grid_to_graph(grid):
	"""
	Transform a binary grid (True == wall) into a
	graph.
	"""
	h, w = grid.shape
	nodes = grid.flatten()
	print('nodes', nodes)
	n = len(nodes)
	A = jnp.zeros((n,n), dtype=jnp.uint32)
	
	all_idx = jnp.arange(n)
	# jax.debug.print('dumneigh idx {x}', x=~nodes)
	dum_neighbor_idx = jnp.argmax(~nodes)
	dum_neighbor_mask = jnp.zeros(n, dtype=jnp.bool_)
	dum_neighbor_mask = \
		dum_neighbor_mask.at[dum_neighbor_idx].set(True)

	def _get_neigbors(idx):
		# Return length n boolean mask of neighbors
		# We then vmap this function over all n
		r = idx + 1
		l = idx - 1
		t = idx - w
		b = idx + w

		l_border = jnp.logical_or(
			idx % w == 0,
			nodes[l]
		)
		r_border = jnp.logical_or(
			r % w == 0,
			nodes[r]
		)
		t_border = jnp.logical_or(
			idx // w == 0,
			nodes[t],
		)
		b_border = jnp.logical_or(
			idx // w == h - 1,
			nodes[b]
		)

		l_ignore = jnp.logical_or(
			l_border,
			nodes[idx]
		)
		r_ignore = jnp.logical_or(
			r_border,
			nodes[idx]
		)
		t_ignore = jnp.logical_or(
			t_border,
			nodes[idx]
		)
		b_ignore = jnp.logical_or(
			b_border,
			nodes[idx]
		)

		left = l*(1-l_ignore) + idx*(l_ignore)
		right = r*(1-r_ignore) + idx*(r_ignore)
		top = t*(1-t_ignore) + idx*(t_ignore)
		bottom = b*(1-b_ignore) + idx*(b_ignore)

		neighbor_mask = jnp.zeros(n, dtype=jnp.bool_)
		# jax.debug.print('idx {x}, neigh {n}', x=idx, n=jnp.array([left, right, top, bottom]))
		neighbor_mask = neighbor_mask.at[jnp.array([left, right, top, bottom])].set(True)

		neighbor_mask = (1-nodes[idx])*neighbor_mask + nodes[idx]*dum_neighbor_mask

		neighbor_mask = neighbor_mask.at[idx].set(False)
		# jax.debug.print('idx {x} mask {m}', x=idx, m=neighbor_mask)

		return neighbor_mask

	A = jax.vmap(_get_neigbors)(all_idx).astype(dtype=jnp.uint32)
	A = jnp.maximum(A, A.transpose())

	return A
	

NEIGHBOR_OFFSETS = jnp.array([
	[1,0], # right
	[0,1], # bottom
	[-1,0], # left
	[0,-1], # top
	[0,0] # self
], dtype=jnp.int32)


@jax.jit
def component_mask_with_pos(grid, pos_a):
	"""
	Return a mask set to True in all cells that are
	a part of the connected component containing pos_a.
	pos_a in format [x, y].
	"""
	h,w = grid.shape
	visited_mask = grid

	pos = pos_a
	visited_mask = visited_mask.at[
		pos[1],pos[0]
	].set(True)
	vstack = jnp.zeros((h*w, 2), dtype=jnp.uint32)
	vstack = vstack.at[:2].set(pos)
	vstack_size = 2

	def _scan_dfs(carry, step):
		(visited_mask, vstack, vstack_size) = carry

		pos = vstack[vstack_size-1]

		neighbors = \
			jnp.minimum(
				jnp.maximum(
					pos + NEIGHBOR_OFFSETS, 0
				), jnp.array([[h,w]])
			).astype(jnp.uint32)

		neighbors_mask = visited_mask.at[
			neighbors[:,1],neighbors[:,0]
		].get()
		n_neighbor_visited = neighbors_mask[:4].sum()
		all_visited = n_neighbor_visited == 4
		all_visited_post = n_neighbor_visited >= 3
		neighbors_mask = neighbors_mask.at[-1].set(~all_visited)

		next_neighbor_idx = jnp.argmax(~neighbors_mask)
		next_neighbor = neighbors[next_neighbor_idx]

		visited_mask = visited_mask.at[
			next_neighbor[1],next_neighbor[0]
		].set(True)

		vstack_size -= all_visited_post

		vstack = vstack.at[vstack_size].set(next_neighbor)
		vstack_size += ~all_visited

		pos = next_neighbor

		return (visited_mask, vstack, vstack_size), None

	max_n_steps = 2*h*w
	(visited_mask, vstack, vstack_size), _ = jax.lax.scan(
		_scan_dfs,
		(visited_mask, vstack, vstack_size),
		jnp.arange(max_n_steps),
		length=max_n_steps
	)

	visited_mask = visited_mask ^ grid
	return visited_mask


@jax.jit
def shortest_path_len(grid, pos_a, pos_b):
    # false should equal free space
	# jax.debug.print('pos_a {x} pos_b {y} grid {g}', x=pos_a, y=pos_b, g=grid)
	grid = ~component_mask_with_pos(grid, pos_a)
	# jax.debug.print('component pos_a {x} pos_b {y} grid {g}', x=pos_a, y=pos_b, g=grid)
	A = grid_to_graph(grid)
	D = apsp(A, n=A.shape[0])
	# jax.debug.print('A {x} {x2}', x=A, x2=A.shape)
	# jax.debug.print('D {x} {x2}', x=D, x2=D.shape)
	# jax.debug.print('pos_b shape {x}', x=pos_b.shape)
	if len(pos_b.shape) == 2: # batch eval
		return jax.vmap(_shortest_path_len, in_axes=(None, None, 0, None))(
			grid, pos_a, pos_b, D
		)
	else:
		return _shortest_path_len(grid, pos_a, pos_b, D)


@jax.jit
def _shortest_path_len(grid, pos_a, pos_b, D):
	h,w = grid.shape

	a_idx = pos_a[1]*w + pos_a[0]
	b_idx = pos_b[1]*w + pos_b[0]
	d = D[a_idx][b_idx]

	mhttn_d = jnp.sum(jnp.abs(jnp.maximum(pos_a,pos_b)- jnp.minimum(pos_a,pos_b)))

	impossible = jnp.logical_and(
		d == 1,
		mhttn_d > 1
	)
	return jax.lax.select(jnp.all(pos_a == pos_b), 1, (d*(1-impossible)).astype(jnp.int32))
	# return d*(1-impossible)
