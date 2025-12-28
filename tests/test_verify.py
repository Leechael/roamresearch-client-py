from roamresearch_client_py.verify import diff_page_against_markdown


def test_verify_diff_empty_when_page_matches_markdown():
    """
    Verification: if fetched page content matches desired markdown, diff is empty.
    """
    page_uid = "page"
    page = {
        ":block/uid": page_uid,
        ":block/children": [
            {
                ":block/uid": "p1",
                ":block/string": "Parent",
                ":block/order": 0,
                ":block/children": [
                    {
                        ":block/uid": "c1",
                        ":block/string": "Child",
                        ":block/order": 0,
                        ":block/children": [],
                    }
                ],
            }
        ],
    }

    markdown = "- Parent\n  - Child\n"
    diff = diff_page_against_markdown(page, markdown)
    assert diff.is_empty()


def test_verify_diff_nonempty_when_page_missing_child():
    """
    Verification: if fetched page content is missing blocks, diff is non-empty.
    """
    page_uid = "page"
    page = {
        ":block/uid": page_uid,
        ":block/children": [
            {
                ":block/uid": "p1",
                ":block/string": "Parent",
                ":block/order": 0,
                ":block/children": [],
            }
        ],
    }

    markdown = "- Parent\n  - Child\n"
    diff = diff_page_against_markdown(page, markdown)
    assert not diff.is_empty()
    assert diff.stats()["creates"] >= 1

