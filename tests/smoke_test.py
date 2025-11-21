def test_package_imports():
    """
    Basic check that the installed wheel can be imported.
    """
    import datastew

    assert datastew is not None


def test_version_exists():
    """
    Ensure version metadata is accessible.
    """
    from datastew import __version__

    assert isinstance(__version__, str)
    assert len(__version__) > 0
