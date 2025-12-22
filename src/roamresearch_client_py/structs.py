from typing import List, Optional, TypeVar
import uuid

class BlockRef:
    db_id: int
    block_uid: int

    def __init__(self, block_uid: str = None, db_id: int = None):
        self.block_uid = block_uid
        self.db_id = db_id

    def is_valid(self):
        return self.block_uid is not None
    
    def is_saved(self):
        return self.db_id is not None

    @classmethod
    def new(cls):
        hex = uuid.uuid4().hex
        return cls(block_uid=hex, db_id=None)
    
    def __repr__(self):
        return f"<BlockRef: {self.block_uid}>"
    
    def __eq__(self, other):
        return self.block_uid == other.block_uid and self.block_uid is not None


class Block:
    ref: BlockRef
    uid: str
    text: str
    order: str | int
    open: bool
    parent_ref: Optional[BlockRef]

    _heading: Optional[int]

    # Metadata after persistence
    create_user: Optional[int] # map to struct {':create/user': {':db/id': __id__ }}
    create_time: Optional[int]
    edit_user: Optional[int] # map to struct {':edit/user': {':db/id': __id__ }}
    edit_time: Optional[int]
    db_id: Optional[int]
    page: Optional[int] # map to struct {':block/page': {':db/id': __id__ }}
    parents: Optional[list['Block'] | list[BlockRef]] # map to struct {':block/parents': [{':db/id': __id__ }, ...]}
    children: Optional[list['Block'] | list[BlockRef]] # map to struct {':block/children': [{...}, ...]}

    def __init__(self, text: str, parent_ref: str | BlockRef | None = None, order: str = "last", open: bool = True, *, ref: BlockRef = None):
        self.ref = ref or BlockRef.new()
        self.text = text
        self.order = order
        self.open = open
        if parent_ref is not None:
            self.parent_ref = parent_ref if isinstance(parent_ref, BlockRef) else BlockRef(block_uid=parent_ref)
        else:
            self.parent_ref = None
        self.parents = []
        self.children = []
        self._heading = None

    @property
    def heading(self):
        return self._heading
    
    @heading.setter
    def heading(self, value):
        if 1 <= value <= 3:
            self._heading = value

    def __repr__(self):
        return f"<Block: {self.ref.block_uid}>"
    
    def __eq__(self, other):
        return self.ref.block_uid == other.ref.block_uid and self.ref.db_id == other.ref.db_id and self.ref is not None

    def to_create_action(self):
        if self.parent_ref is None:
            raise ValueError("Parent reference is required for create action")
        dic = {
            "action": "create-block",
            "location": {
                "parent-uid": self.parent_ref.block_uid,
                "order": self.order or "last",
            },
            "block": {
                "uid": self.ref.block_uid,
                "string": self.text,
                "open": self.open or True,
            },
        }
        if self._heading:
            assert self._heading <= 3
            dic["block"]["heading"] = self._heading
        return dic

    def to_update_action(self, include_text: bool = True) -> dict:
        """
        Generate update-block action.

        Can update: string, heading, open
        Cannot update: order, parent (use to_move_action for those)
        """
        action = {
            "action": "update-block",
            "block": {"uid": self.ref.block_uid}
        }
        if include_text and self.text is not None:
            action["block"]["string"] = self.text
        if self._heading is not None:
            action["block"]["heading"] = self._heading
        if self.open is not None:
            action["block"]["open"] = self.open
        return action

    def to_move_action(self, parent_uid: str, order: int | str = "last") -> dict:
        """
        Generate move-block action.

        Use this to change parent or order of a block while preserving its UID.
        """
        return {
            "action": "move-block",
            "block": {"uid": self.ref.block_uid},
            "location": {
                "parent-uid": parent_uid,
                "order": order
            }
        }

    def to_delete_action(self) -> dict:
        """
        Generate delete-block action.

        Warning: This will recursively delete all children blocks.
        """
        return {
            "action": "delete-block",
            "block": {"uid": self.ref.block_uid}
        }
    
    @classmethod
    def from_dict(cls, action: dict):
        # TODO: handle recursive parents & children
        # TODO: based on query, we can get the full block object or partial object
        obj = cls(
            text=action.get(":block/string", ""),
            order=action.get(":block/order"),
            open=action.get(":block/open"),
            ref=BlockRef(block_uid=action.get(":block/uid"), db_id=action.get(":db/id"))
        )
        if ":create/user" in action:
            obj.create_user = action[":create/user"][":db/id"]
        if ":create/time" in action:
            obj.create_time = action[":create/time"]
        if ":edit/user" in action:
            obj.edit_user = action[":edit/user"][":db/id"]
        if ":edit/time" in action:
            obj.edit_time = action[":edit/time"]
        if ":block/page" in action:
            obj.page = action[":block/page"][":db/id"]
        if ":block/parents" in action:
            obj.parents = [BlockRef(block_uid=p.get(":block/uid"), db_id=p.get(":db/id"))
                           for p in action[":block/parents"]]
        else:
            obj.parents = []
        if ":block/children" in action:
            obj.children = [BlockRef(block_uid=c.get(":block/uid"), db_id=c.get(":db/id"))
                            for c in action[":block/children"]]
        else:
            obj.children = []
        return obj