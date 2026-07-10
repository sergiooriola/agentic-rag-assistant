# HTTP Essentials

A practical reference on the Hypertext Transfer Protocol (HTTP) for web developers.

## Request methods

HTTP defines a set of request methods, each with well-defined semantics:

- **GET** retrieves a representation of a resource. It must be *safe* (no side effects)
  and *idempotent* (repeating the request yields the same result).
- **POST** submits data to a resource and typically creates a new subordinate resource.
  POST is neither safe nor idempotent: sending the same POST twice may create two records.
- **PUT** replaces the target resource entirely with the request payload. PUT is idempotent:
  applying it multiple times leaves the resource in the same state.
- **PATCH** applies a partial modification to a resource. Unlike PUT, it sends only the
  fields that change, and it is not required to be idempotent.
- **DELETE** removes the target resource and is idempotent.
- **HEAD** is identical to GET but returns only the headers, useful for checking existence
  or freshness without transferring the body.

## Status code classes

Status codes are grouped into five classes by their first digit:

- **1xx (Informational)**: the request was received and the process continues.
- **2xx (Success)**: the request succeeded. `200 OK` is the generic success; `201 Created`
  signals that a new resource was created; `204 No Content` indicates success with an
  empty response body.
- **3xx (Redirection)**: further action is needed. `301 Moved Permanently` and
  `302 Found` redirect the client; `304 Not Modified` tells a caching client that its
  stored copy is still valid.
- **4xx (Client Error)**: the client made a mistake. `400 Bad Request` covers malformed
  input, `401 Unauthorized` means missing or invalid authentication, `403 Forbidden`
  means authenticated but not allowed, `404 Not Found` means the resource does not exist,
  and `429 Too Many Requests` signals rate limiting.
- **5xx (Server Error)**: the server failed to fulfill a valid request. `500 Internal
  Server Error` is the generic failure and `503 Service Unavailable` indicates temporary
  overload or maintenance.

## Headers and content negotiation

Headers carry metadata about the request or response. `Content-Type` declares the media
type of the body (for example `application/json`), while `Accept` lets the client state
which media types it can process. This mechanism is called *content negotiation*: the
server picks a representation compatible with the client's `Accept` header and reports
its choice in `Content-Type`.

Authentication commonly travels in the `Authorization` header, for example
`Authorization: Bearer <token>` for token-based schemes.

## Caching fundamentals

HTTP caching reduces latency and server load. The `Cache-Control` header is the primary
control: `max-age` defines how many seconds a response stays fresh, `no-store` forbids
caching entirely, and `private` restricts caching to the end user's browser.

Validation allows a cache to reuse stale entries cheaply: the server sends an `ETag`
(an opaque version identifier) and the client revalidates with `If-None-Match`. If the
resource is unchanged, the server answers `304 Not Modified` with no body, saving
bandwidth.

## HTTPS in one paragraph

HTTPS is HTTP over TLS. TLS provides three guarantees: *confidentiality* (traffic is
encrypted), *integrity* (tampering is detectable), and *authentication* (the server
proves its identity with a certificate issued by a trusted authority). Modern browsers
treat plain HTTP as insecure, so HTTPS is the default for any production deployment.
