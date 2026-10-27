"""Public test-only TLS material for verified localhost connections.

The certificate is valid from 2020 through 2120 and contains localhost/127.0.0.1
SANs. This deliberately public private key is exclusively a local test fixture.
"""

import ssl

_CERTIFICATE = """-----BEGIN CERTIFICATE-----
MIIC5DCCAcygAwIBAgIBATANBgkqhkiG9w0BAQsFADAUMRIwEAYDVQQDDAlsb2Nh
bGhvc3QwIBcNMjAwMTAxMDAwMDAwWhgPMjEyMDAxMDEwMDAwMDBaMBQxEjAQBgNV
BAMMCWxvY2FsaG9zdDCCASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBAPVa
ji3V1CtNsFIzHUslkd5+SG1xpKXIRfigqknxUsr1vHVHb9l1jurDkFJraOUW5jdx
E/R/VR7zq4KBPYyJODnrxyO/p9t6nRhqjTrEnSIWmJ/M2ys5QGFsewyceOCG7Y+E
6sf1J9ii63fhuZjXj2NXbADT+jPG2c5XzjdvPrGCzkDOBxtlPHJ+HhB0zxoMsyhU
forSxrUB1M+8CfJarVg2KoKA1O8pkirKWzpptOFHujDhCvej1P52DrDuGTOazhov
aAUcUYmCorksgQxO3lANxnAvOonKDYF3kg2QGVXgygoS/DjomZ5mOTh+nJntZ1Yf
iuBJim03qO2LsZaT880CAwEAAaM/MD0wDwYDVR0TAQH/BAUwAwEB/zAOBgNVHQ8B
Af8EBAMCAYYwGgYDVR0RBBMwEYIJbG9jYWxob3N0hwR/AAABMA0GCSqGSIb3DQEB
CwUAA4IBAQBIUxnSRoI+vEfSBiJzhylC9yvQg/P0wVntL8LWJCIwWQ+v5f0FCoB2
jxd98sQkEe7vnKXHNKk63MBRq9dkS78oVl5EIbrdHBzp71aAM6FSRG3WHINxxofO
T8tpqLs2y0ZoTRgJjkkvY7ZvTT1AGg7g8tV+U+AU70W1e2EouWJpzocNotM4B0rt
Jt8hEAcsaoS4WnLwvOOrB9dET5DOWH+FJDhA6Caj24v39VouQVKOZ56Ud1G23IAv
A32TKNEdCSJkigHyzUiV80V7vir8+Iq7o/wrn242hAf8WYNObRSpO8xpw+eJBbdk
3QRLJJwfWpiycZXbPJEVxDesfkh86v1D
-----END CERTIFICATE-----
"""
_PRIVATE_KEY = """-----BEGIN PRIVATE KEY-----
MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQD1Wo4t1dQrTbBS
Mx1LJZHefkhtcaSlyEX4oKpJ8VLK9bx1R2/ZdY7qw5BSa2jlFuY3cRP0f1Ue86uC
gT2MiTg568cjv6fbep0Yao06xJ0iFpifzNsrOUBhbHsMnHjghu2PhOrH9SfYout3
4bmY149jV2wA0/ozxtnOV843bz6xgs5AzgcbZTxyfh4QdM8aDLMoVH6K0sa1AdTP
vAnyWq1YNiqCgNTvKZIqyls6abThR7ow4Qr3o9T+dg6w7hkzms4aL2gFHFGJgqK5
LIEMTt5QDcZwLzqJyg2Bd5INkBlV4MoKEvw46JmeZjk4fpyZ7WdWH4rgSYptN6jt
i7GWk/PNAgMBAAECggEADUPsyqme+QsQQhbQD09kCEWHoPMzVbXy992n7KuG2xZX
hCJtTQfqxuKf8PTXDQZ57VFzJMBQV9tmJdOFrQ2A5BDNXtoZNz1KimwTCrRf/THU
5VRBcqAmd539Nnysmbxx0lVQuahAPpmVuuY8ufjISzS00C6l08V9N9zOIiJ6Se/k
y8qJ8wlt0J3Bp7E4bcajRV7CzRBWr/tk9XuXDP1rB0iSJLANOXANqqhoYYeFts/w
+wgHPEX0mRsqYbYXQv+UgOsTgvSs/V+WXk1ZLj1ZEOD/ZEL/MNBsp3rN9Dr94FSH
dlrNNYi7Tl11Y7L4F/FAnkaCCB5nf3UviEIpkiWKOQKBgQD7KbbLEGaWo0Ct4yVu
Vwa8N1YlF0FwcTetcy+JY/8lHo1ryJd6mL9CpzPSudPX5p75SIau0PKSOC9FveDI
+tgE/JhGTrt0yyu8qYsyeshZTsFJo+BVDpH4rTQV9l03UUYwpgCOWOymMbNkLJGT
1h7XomFy44DO0s2yGl0qOv7E0wKBgQD6FDNhyYjSQ1mYb2DOHztBpUZkiswqRDed
QXCsDkyyIHMInhsdQJmshL0WQhLPqUjGR7Ix1wlv7N/LulO1lGhFM8bhgWH5Wa4M
gVJBpEWYKJjoHca5R2u0gUPpNQqPgt1fJJ5uCxi9yQZeecQDHJ+PNNdbwmA9warq
iOJkLK+A3wKBgGvILuVdmGi0jiERBRMgzYcfyBwKqs5yB46ri47OFsP7JH1Ii5jU
y97ymc6oaDySf05CzWlfyXjuG3s2TKxy2RjqvP4O82rfdVl0+IMlJa0NmuzbIn3d
zs6EpYF3Q/FlM1XVJw/vIT8AvRcT+TUUAhBPYrpG3xPiatsvbSOeRY55AoGAP4es
OfZh3NChTTlWRWbnvdiH1njwPviPjFKYlg0X7NSThKftyXOk1o/uHV5xuvGNfFmw
ajrvudBaVj4ij+mAx7vMlXMx+Y+PyLt9sxFRYMNHfQg7eMwtD3EbWH5SDX04UbhS
yy/kfCo6hPu4EPZPjqjk4PmzS10M14IoyTwU5c8CgYEAj5xHKDJKDRme2K34w6er
KRXC6RKr/0p1gW0W2CIqNY1u74eUsthRIF47W6xp26ASSChv3y1xIjUdnTXr+kUM
ZyE4KSiZON31Fwi+ZQNMEN10THInlfiBxnglv4s0NUEFgn+cPbzT/0s3s5qx4xj1
j1UFeMba4bEmVoEbQTUKETU=
-----END PRIVATE KEY-----
"""


def tls_contexts(tmp_path):
    """Return a server context and a client that trusts only the test issuer."""
    certificate = tmp_path / "localhost.pem"
    private_key = tmp_path / "localhost-key.pem"
    certificate.write_text(_CERTIFICATE, encoding="ascii")
    private_key.write_text(_PRIVATE_KEY, encoding="ascii")
    server = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server.load_cert_chain(certificate, private_key)
    client = ssl.create_default_context(cafile=str(certificate))
    assert client.check_hostname
    assert client.verify_mode == ssl.CERT_REQUIRED
    return server, client
