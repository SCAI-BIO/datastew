def test_package_imports():
    """
    Basic check that the installed wheel and its core submodules can be imported.
    This helps catch missing third-party dependencies in the build configuration.
    """
    import datastew

    assert datastew is not None

    from datastew import embedding, harmonization, io, repository, visualisation

    assert all([embedding, harmonization, io, repository, visualisation])


def test_version_exists():
    """
    Ensure version metadata is accessible.
    """
    from datastew import __version__

    assert isinstance(__version__, str)
    assert len(__version__) > 0
