mongo_ip = "mongodb://localhost:27017/LendingClub."

DROP_COLUMNS = ["emp_title",  # Too many job title can't convert into a dummy variable
                "emp_length",  # Bad attribute
                "title",  # title column is basically a string subcategory/description.
                "grade", #
                "issue_d"]
