import jax.numpy as jnp
import jax


def downsample(img, factor):
    """
    Downsample an image along both dimensions by some factor
    """
    assert img.shape[0] % factor == 0
    assert img.shape[1] % factor == 0

    img = img.reshape(
        [img.shape[0] // factor, factor, img.shape[1] // factor, factor, 3]
    )
    img = img.mean(axis=3)
    img = img.mean(axis=1)

    # convert back to uint8
    img = img.astype(jnp.uint8)

    return img


def fill_coords(img, fn, color):
    """
    Fill pixels of an image with coordinates matching a filter function
    """

    def _mask_fn(y, x):
        yf = (y + 0.5) / img.shape[0]
        xf = (x + 0.5) / img.shape[1]
        return fn(xf, yf)

    ys, xs = jnp.indices(img.shape[:2])
    mask = jax.vmap(jax.vmap(_mask_fn))(ys, xs)

    color_img = jnp.full_like(img, color)
    return jnp.where(mask[:, :, None], color_img, img)


def rotate_fn(fin, cx, cy, theta):
    def fout(x, y):
        x = x - cx
        y = y - cy

        x2 = cx + x * jnp.cos(-theta) - y * jnp.sin(-theta)
        y2 = cy + y * jnp.cos(-theta) + x * jnp.sin(-theta)

        return fin(x2, y2)

    return fout


def point_in_line(x0, y0, x1, y1, r):
    p0 = jnp.array([x0, y0])
    p1 = jnp.array([x1, y1])
    dir = p1 - p0
    dist = jnp.linalg.norm(dir)
    dir = dir / dist

    xmin = min(x0, x1) - r
    xmax = max(x0, x1) + r
    ymin = min(y0, y1) - r
    ymax = max(y0, y1) + r

    def fn(x, y):
        # Fast, early escape test
        if x < xmin or x > xmax or y < ymin or y > ymax:
            return False

        q = jnp.array([x, y])
        pq = q - p0

        # Closest point on line
        a = jnp.dot(pq, dir)
        a = jnp.clip(a, 0, dist)
        p = p0 + a * dir

        dist_to_line = jnp.linalg.norm(q - p)
        return dist_to_line <= r

    return fn


def point_in_circle(cx, cy, r):
    def fn(x, y):
        return (x - cx) * (x - cx) + (y - cy) * (y - cy) <= r * r

    return fn


def point_in_rect(xmin, xmax, ymin, ymax):
    def fn(x, y):
        return (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)

    return fn


def point_in_triangle(a, b, c):
    a = jnp.array(a)
    b = jnp.array(b)
    c = jnp.array(c)

    def fn(x, y):
        v0 = c - a
        v1 = b - a
        v2 = jnp.array((x, y)) - a

        # Compute dot products
        dot00 = jnp.dot(v0, v0)
        dot01 = jnp.dot(v0, v1)
        dot02 = jnp.dot(v0, v2)
        dot11 = jnp.dot(v1, v1)
        dot12 = jnp.dot(v1, v2)

        # Compute barycentric coordinates
        inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom

        # Check if point is in triangle
        return (u >= 0) & (v >= 0) & (u + v < 1)

    return fn


def highlight_img(img, color=(255, 255, 255), alpha=0.30):
    """
    Add highlighting to an image
    """
    blend_img = img + alpha * (jnp.array(color, dtype=jnp.uint8) - img)
    res_img = jnp.clip(blend_img, 0, 255).astype(jnp.uint8)

    return res_img
