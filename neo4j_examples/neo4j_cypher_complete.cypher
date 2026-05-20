// ============================================================
// Neo4j Cypher 语法大全 - 循序渐进教程
// 每条语句可依次执行，后面的复杂查询依赖前面创建的数据
// ============================================================

// ============================================================
// 第一部分：清理环境 (执行前先用此语句清空数据库)
// ============================================================

// 清空所有节点和关系 (谨慎使用)
// MATCH - 匹配模式子句，用于查找图中符合条件的节点或关系
// DETACH - 分离关键字，先断开节点的所有关系
// DELETE - 删除关键字，删除节点或关系本身
// 组合含义：先匹配所有节点 (n)，然后分离并删除它们（包括所有关系）
MATCH (n) DETACH DELETE n;

// ============================================================
// 第二部分：创建节点 (CREATE)
// ============================================================

// --- 2.1 创建单个节点 ---
// CREATE - 创建关键字，用于创建新的节点或关系
// (p:Person {...}) - 创建标签为 Person 的节点，变量名为 p，花括号内是属性键值对
CREATE (p:Person {name: 'Alice', age: 30, user_id: 1007})
// RETURN - 返回关键字，指定查询结果要返回的内容
// 如果只是执行 CREATE、MERGE、DELETE、SET 等写操作（不关心返回结果），可以不加 RETURN。
// 但如果希望看到创建/修改的结果（比如调试或获取新节点数据），就必须用 RETURN。
RETURN p;

// --- 2.2 创建多个节点 ---
CREATE (p1:Person {name: 'Bob', age: 25}),
       (p2:Person {name: 'Charlie', age: 35}),
       (p3:Person {name: 'David', age: 28})
RETURN p1, p2, p3;

// --- 2.3 创建带有多属性的节点 ---
CREATE (m:Movie {
    title: 'The Matrix',
    released: 1999,
    tagline: 'Welcome to the Real World',
    rating: 8.7
})
RETURN m;

// --- 2.4 创建更多电影节点 ---
CREATE (m1:Movie {title: 'Inception', released: 2010, rating: 8.8}),
       (m2:Movie {title: 'Interstellar', released: 2014, rating: 8.6}),
       (m3:Movie {title: 'The Dark Knight', released: 2008, rating: 9.0})
RETURN m1, m2, m3;

// --- 2.5 创建公司节点 ---
CREATE (c1:Company {name: 'Warner Bros', founded: 1923}),
       (c2:Company {name: 'Paramount', founded: 1912}),
       (c3:Company {name: 'Netflix', founded: 1997})
RETURN c1, c2, c3;

// --- 2.6 创建导演节点 ---
CREATE (d1:Director {name: 'Lana Wachowski', birthYear: 1965}),
       (d2:Director {name: 'Christopher Nolan', birthYear: 1970}),
       (d3:Director {name: 'Tim Burton', birthYear: 1958})
RETURN d1, d2, d3;

// ============================================================
// 第三部分：创建关系 (CREATE Relationship)
// ============================================================

// --- 3.1 创建 ACTED_IN 关系 (演员出演电影) ---
//MATCH 是图数据库查询语言 Cypher 中的一个核心关键字，它的作用是在图中查找并匹配已经存在的节点或关系模式。
//MATCH：先找到已经存在的 Alice 节点和 The Matrix 节点。
//CREATE：在找到的这两个节点之间，创建一条从 Alice 指向 The Matrix 的 ACTED_IN 类型的边。
//:ACTED_IN：这是要创建的关系（边）的类型。
//{roles: ['Neo']}：在创建这条边的同时，为它设置一个 roles 属性，值为 ['Neo']。
MATCH (p:Person {name: 'Alice'}), (m:Movie {title: 'The Matrix'})
CREATE (p)-[:ACTED_IN {roles: ['Neo']}]->(m)
RETURN p, m;

// --- 3.2 创建带多个角色的关系 ---
MATCH (p:Person {name: 'Bob'}), (m:Movie {title: 'Inception'})
CREATE (p)-[:ACTED_IN {roles: ['Cobb']}]->(m)
RETURN p, m;

// --- 3.3 创建多个人出演同一部电影 ---
MATCH (p1:Person {name: 'Charlie'}),
      (p2:Person {name: 'David'}),
      (m:Movie {title: 'Interstellar'})
CREATE (p1)-[:ACTED_IN {roles: ['Cooper']}]->(m),
       (p2)-[:ACTED_IN {roles: ['TARS']}]->(m)
RETURN p1, p2, m;

// --- 3.4 创建 DIRECTED 关系 (导演执导电影) ---
MATCH (d:Director {name: 'Lana Wachowski'}), (m:Movie {title: 'The Matrix'})
CREATE (d)-[:DIRECTED {year: 1999}]->(m)
RETURN d, m;

// --- 3.5 创建更多导演关系 ---
MATCH (d:Director {name: 'Christopher Nolan'}),
      (m1:Movie {title: 'Inception'}),
      (m2:Movie {title: 'Interstellar'}),
      (m3:Movie {title: 'The Dark Knight'})
CREATE (d)-[:DIRECTED {year: 2010}]->(m1),
       (d)-[:DIRECTED {year: 2014}]->(m2),
       (d)-[:DIRECTED {year: 2008}]->(m3)
RETURN d, m1, m2, m3;

// --- 3.6 创建 PRODUCED_BY 关系 (电影由公司制作) ---
MATCH (m:Movie), (c:Company)
WHERE (m.title = 'The Matrix' AND c.name = 'Warner Bros')
   OR (m.title = 'Inception' AND c.name = 'Warner Bros')
   OR (m.title = 'Interstellar' AND c.name = 'Paramount')
   OR (m.title = 'The Dark Knight' AND c.name = 'Warner Bros')
CREATE (m)-[:PRODUCED_BY]->(c)
RETURN m.title, c.name;

// ============================================================
// 第四部分：查询节点 (MATCH + RETURN)
// ============================================================

// --- 4.1 查询所有节点 ---
MATCH (n) RETURN n;

// --- 4.2 查询指定标签的节点 ---
MATCH (p:Person) RETURN p;

// --- 4.3 查询指定属性的节点 ---
MATCH (p:Person {name: 'Alice'}) RETURN p;

// --- 4.4 查询并返回特定属性 ---
MATCH (p:Person) RETURN p.name, p.age;

// --- 4.5 查询时重命名返回列 (使用反引号包裹中文别名)
// AS - 别名关键字，用于给返回列或变量起别名
MATCH (p:Person)
RETURN p.name AS `姓名`, p.age AS `年龄`;

// --- 4.6 查询满足条件的节点 ---
// WHERE - 条件子句，用于过滤 MATCH 匹配的结果，类似 SQL 中的 WHERE
MATCH (p:Person)
WHERE p.age > 27
RETURN p.name AS `姓名`, p.age AS `年龄`;

// --- 4.7 查询电影并按评分排序 ---
// ORDER BY - 排序子句，用于对结果进行排序
// DESC - 降序排列 (从大到小)，ASC 为升序 (默认)
MATCH (m:Movie)
RETURN m.title AS `电影名`, m.rating AS `评分`
ORDER BY m.rating DESC;

// --- 4.8 限制返回结果数量 ---
// LIMIT - 限制关键字，用于限制返回结果的数量
// SKIP - 跳过关键字，用于跳过指定数量的结果（常用于分页）
MATCH (m:Movie)
RETURN m.title, m.rating
ORDER BY m.rating DESC
LIMIT 2;

// --- 4.9 跳过前 N 条结果 ---
// 先跳过 1 条，再限制返回 2 条，实现分页效果
MATCH (m:Movie)
RETURN m.title, m.rating
ORDER BY m.rating DESC
SKIP 1 LIMIT 2;

// ============================================================
// 第五部分：条件查询 (WHERE)
// ============================================================

// --- 5.1 等于条件 ---
MATCH (p:Person)
WHERE p.age = 30
RETURN p.name;

// --- 5.2 不等于条件 ---
MATCH (p:Person)
WHERE p.age <> 30
RETURN p.name, p.age;

// --- 5.3 范围条件 ---
// AND - 逻辑与操作符，连接多个条件，所有条件都需满足
MATCH (p:Person)
WHERE p.age >= 25 AND p.age <= 30
RETURN p.name, p.age;

// --- 5.4 IN 操作符 ---
// IN - 属于操作符，检查值是否在列表中
MATCH (m:Movie)
WHERE m.title IN ['The Matrix', 'Inception']
RETURN m.title, m.released;

// --- 5.5 LIKE 模糊匹配 ---
// CONTAINS - 包含操作符，检查字符串是否包含子串
MATCH (m:Movie)
WHERE m.title CONTAINS 'The'
RETURN m.title;

// --- 5.6 正则表达式匹配 ---
// =~ - 正则匹配操作符，使用正则表达式进行模式匹配
MATCH (p:Person)
WHERE p.name =~ 'A.*'
RETURN p.name;

// --- 5.7 检查属性是否存在 ---
// exists() - 函数，检查属性或模式是否存在
MATCH (n)
WHERE exists(n.released)
RETURN labels(n), count(n);

// --- 5.8 OR 条件 ---
// OR - 逻辑或操作符，连接多个条件，满足任一条件即可
MATCH (p:Person)
WHERE p.age < 26 OR p.age > 34
RETURN p.name, p.age;

// ============================================================
// 第六部分：聚合函数 (Aggregation)
// ============================================================

// --- 6.1 COUNT - 统计节点数量 ---
// count() - 聚合函数，统计匹配的行数或节点数
MATCH (p:Person)
RETURN count(p) AS `总人数`;

// --- 6.2 COUNT DISTINCT - 统计不重复的值 ---
// DISTINCT - 去重关键字，只统计不重复的值
MATCH (p:Person)
RETURN count(DISTINCT p.age) AS `不同年龄数`;

// --- 6.3 SUM - 求和 ---
// sum() - 聚合函数，计算数值的总和
MATCH (p:Person)
RETURN sum(p.age) AS `年龄总和`;

// --- 6.4 AVG - 平均值 ---
// avg() - 聚合函数，计算数值的平均值
MATCH (p:Person)
RETURN avg(p.age) AS `平均年龄`;

// --- 6.5 MAX/MIN - 最大/最小值 ---
// max() - 聚合函数，返回最大值
// min() - 聚合函数，返回最小值
MATCH (m:Movie)
RETURN max(m.rating) AS `最高评分`,
       min(m.rating) AS `最低评分`;

// --- 6.6 GROUP BY - 按标签分组统计 ---
// labels() - 函数，返回节点的所有标签列表
MATCH (n)
RETURN labels(n) AS `类型`, count(n) AS `数量`
ORDER BY count(n) DESC;

// --- 6.7 COLLECT - 收集值为列表 ---
// collect() - 聚合函数，将值收集为一个列表（数组）
MATCH (p:Person)
RETURN collect(p.name) AS `所有人`;

// ============================================================
// 第七部分：关系查询
// ============================================================

// --- 7.1 查询有关系的节点对 ---
// -[r:ACTED_IN]-> - 关系模式，r 是关系变量，ACTED_IN 是关系类型
// 箭头方向表示关系方向，从 Person 指向 Movie
MATCH (p:Person)-[r:ACTED_IN]->(m:Movie)
RETURN p.name AS `演员`, m.title AS `电影`;

// --- 7.3 多跳关系查询 ---
// <-[:DIRECTED]- - 反向关系，从 Movie 指向 Director
// 这表示导演执导电影的逆关系
MATCH (p:Person)-[:ACTED_IN]->(m:Movie)<-[:DIRECTED]-(d:Director)
RETURN p.name AS `演员`, m.title AS `电影`, d.name AS `导演`;

// --- 7.4 可变长度关系 - 查询 1 到 3 度关系 ---
// [*1..3] - 可变长度关系，匹配 1 到 3 跳的任意关系
// path = - 将整个路径赋值给变量 path
MATCH path = (p:Person {name: 'Alice'})-[*1..3]-(other)
RETURN path LIMIT 5;

// ============================================================
// 第八部分： OPTIONAL MATCH (左连接)
// ============================================================

// --- 8.1 OPTIONAL MATCH - 匹配可选关系 ---
// OPTIONAL MATCH - 可选匹配子句，类似 SQL 的 LEFT JOIN
// 如果找不到匹配的关系，返回 null 而不是排除该行
MATCH (p:Person)
OPTIONAL MATCH (p)-[:DIRECTED]->(m)
RETURN p.name, m.title;

// --- 8.2 OPTIONAL MATCH with WHERE ---
MATCH (p:Person {name: 'Alice'})
OPTIONAL MATCH (p)-[r:ACTED_IN]->(m:Movie)
WHERE m.rating > 8.5
RETURN p.name, m.title, m.rating;

// ============================================================
// 第九部分：UNION 合并查询结果
// ============================================================

// --- 9.1 UNION - 合并两个查询结果 (去重) ---
// UNION - 合并操作符，将两个查询结果合并为一个，自动去重
// 注意：两个查询的返回列数量和类型必须一致
MATCH (p:Person) RETURN p.name AS name
UNION
MATCH (d:Director) RETURN d.name AS name;

// --- 9.2 UNION ALL - 合并两个查询结果 (保留重复) ---
// UNION ALL - 合并操作符，保留所有结果包括重复项
MATCH (p:Person) RETURN p.name AS name
UNION ALL
MATCH (d:Director) RETURN d.name AS name;

// ============================================================
// 第十部分：WITH 子句 (管道传递)
// ============================================================

// --- 10.1 WITH - 传递中间结果 ---
// WITH - 子句，将前面查询的结果传递给后面的查询
// 类似管道操作，可以在 WITH 中进行变量重命名或聚合
MATCH (p:Person)
WITH p, p.age AS age
WHERE age > 27
RETURN p.name, age;

// --- 10.2 WITH + 聚合 ---
// WITH 可以与聚合函数一起使用，实现分组统计
MATCH (p:Person)-[:ACTED_IN]->(m:Movie)
WITH p, count(m) AS movieCount
WHERE movieCount >= 1
RETURN p.name, movieCount
ORDER BY movieCount DESC;

// ============================================================
// 第十一部分：更新数据 (SET)
// ============================================================

// --- 11.1 SET - 更新单个属性 ---
// SET - 更新关键字，用于修改节点的属性或添加标签
MATCH (p:Person {name: 'Alice'})
SET p.age = 31
RETURN p;

// --- 11.3 SET - 添加新标签 ---
// SET p:VIP - 给节点添加 VIP 标签
MATCH (p:Person {name: 'Alice'})
SET p:VIP
RETURN p, labels(p);

// --- 11.4 REMOVE - 删除属性 ---
// REMOVE - 删除关键字，用于删除节点属性或标签
MATCH (p:Person {name: 'Bob'})
REMOVE p.city
RETURN p;

// ============================================================
// 第十二部分：删除数据 (DELETE)
// ============================================================

// --- 12.1 DELETE - 删除节点 (需要先删除关系) ---
// 先创建一个测试节点
CREATE (temp:TempNode {name: 'ToDelete'})
RETURN temp;

// 删除刚创建的节点 (该节点没有关系，可以直接 DELETE)
MATCH (n:TempNode {name: 'ToDelete'})
DELETE n;

// ============================================================
// 第十三部分：MERGE (不存在则创建)
// ============================================================

// --- 13.1 MERGE - 查找或创建节点 ---
// MERGE - 合并关键字，查找匹配的节点，不存在则创建
// 类似"查找或创建"操作，保证数据唯一性
MERGE (p:Person {name: 'Eve'})
RETURN p;

// --- 13.2 MERGE - 创建时设置默认属性 ---
// ON CREATE SET - 仅在节点被创建时执行 SET 操作
MERGE (p:Person {name: 'Frank'})
ON CREATE SET p.age = 30, p.created = timestamp()
RETURN p;

// --- 13.3 MERGE - 更新已存在节点 ---
// ON MATCH SET - 仅在节点已存在时执行 SET 操作
MERGE (p:Person {name: 'Eve'})
ON MATCH SET p.lastSeen = timestamp()
RETURN p;

// ============================================================
// 第十四部分：索引和约束 (Indexes & Constraints)
// ============================================================

// 注意：以下约束和索引语句需要在 Neo4j 5.x 中执行
// Neo4j 4.x 语法略有不同，如果报错请移除 IF NOT EXISTS

// --- 14.1 创建唯一约束 ---
// CONSTRAINT - 约束关键字，用于保证数据完整性
// FOR (p:Person) REQUIRE p.name IS UNIQUE - 要求 Person 的 name 属性唯一
//CREATE CONSTRAINT person_name_unique FOR (p:Person) REQUIRE p.name IS UNIQUE;

// --- 14.2 创建属性存在约束 ---
// IS NOT NULL - 非空约束，要求属性必须存在
//CREATE CONSTRAINT movie_title_required FOR (m:Movie) REQUIRE m.title IS NOT NULL;

// --- 14.3 创建索引 ---
// INDEX - 索引关键字，用于加速查询
//CREATE INDEX person_age_index FOR (p:Person) ON (p.age);

// --- 14.4 创建复合索引 ---
// 在多个属性上创建索引，加速组合查询
//CREATE INDEX movie_title_rating_index FOR (m:Movie) ON (m.title, m.rating);

// --- 14.5 查看所有约束 ---
// SHOW - 显示关键字，查看数据库中的约束或索引
//SHOW CONSTRAINTS;

// --- 14.6 查看所有索引 ---
//SHOW INDEXES;

// --- 14.7 删除约束 ---
// DROP - 删除关键字，删除约束或索引
// 注意：如果约束不存在，这条语句会报错，可以注释掉或跳过
//DROP CONSTRAINT person_name_unique;

// --- 14.8 删除索引 ---
DROP INDEX person_age_index;

// ============================================================
// 第十五部分：复杂查询示例
// ============================================================

// 注意：以下复杂查询依赖前面创建的数据
// 如果数据已被清空，请先执行第 17 部分的完整示例

// --- 15.1 查询合作过的演员 (共同出演同一电影) ---
MATCH (p1:Person)-[:ACTED_IN]->(m:Movie)<-[:ACTED_IN]-(p2:Person)
WHERE p1.name < p2.name
RETURN p1.name AS `演员 1`, p2.name AS `演员 2`, m.title AS `共同电影`
ORDER BY p1.name, p2.name;

// --- 15.2 查询导演执导的所有电影及平均评分 ---
MATCH (d:Director)-[:DIRECTED]->(m:Movie)
RETURN d.name AS `导演`,
       collect(m.title) AS `电影列表`,
       count(m) AS `电影数量`,
       avg(m.rating) AS `平均评分`
ORDER BY avg(m.rating) DESC;

// --- 15.3 查询出演电影最多的演员 ---
MATCH (p:Person)-[:ACTED_IN]->(m:Movie)
RETURN p.name AS `演员`, count(m) AS `电影数量`
ORDER BY count(m) DESC
LIMIT 5;

// --- 15.4 查询某公司的所有电影 ---
MATCH (c:Company {name: 'Warner Bros'})<-[:PRODUCED_BY]-(m:Movie)
RETURN c.name AS `公司`,
       collect(m.title) AS `电影列表`,
       avg(m.rating) AS `平均评分`;

// --- 15.5 最短路径查询 - 查找两人之间的最短关系路径 ---
// shortestPath() - 函数，查找两个节点之间的最短路径
// [*] - 任意长度、任意类型的关系
MATCH path = shortestPath(
    (p1:Person {name: 'Alice'})-[*]-(p2:Person {name: 'Charlie'})
)
RETURN path;

// --- 15.6 所有路径查询 - 查找两人之间的所有路径 ---
// allShortestPaths() - 函数，查找所有最短路径（可能有多条）
// [*..5] - 最多 5 跳的任意关系
MATCH path = allShortestPaths(
    (p1:Person {name: 'Alice'})-[*..5]-(p2:Person)
)
WHERE p2.name IN ['Bob', 'Charlie']
RETURN path LIMIT 10;

// ============================================================
// 第十六部分：高级函数
// ============================================================

// --- 16.1 size - 获取关系或属性数量 ---
// size() - 函数，返回列表或路径的长度（元素数量）
MATCH (p:Person)
RETURN p.name, size(labels(p)) AS `标签数量`;

// --- 16.2 head 和 last - 获取列表首尾元素 ---
// head() - 函数，返回列表的第一个元素
// last() - 函数，返回列表的最后一个元素
MATCH (p:Person)
WITH collect(p.name) AS names
RETURN head(names) AS `第一个`, last(names) AS `最后一个`;

// --- 16.3 reduce - 列表归约 ---
// reduce() - 函数，将列表归约为单个值
// 语法：reduce(变量 = 初始值，元素 IN 列表 | 表达式)
MATCH (p:Person)
WITH collect(p.age) AS ages
RETURN reduce(sum = 0, age IN ages | sum + age) AS `年龄总和`;

// --- 16.4 filter - 列表过滤 ---
// filter() - 函数，过滤列表中满足条件的元素
// 语法：filter(变量 IN 列表 WHERE 条件)
MATCH (m:Movie)
WITH collect(m) AS movies
RETURN filter(m IN movies WHERE m.rating > 8.7) AS `高分电影`;

// --- 16.5 CASE WHEN - 条件表达式 ---
// CASE WHEN - 条件表达式，类似编程中的 if-else
// WHEN - 条件分支
// ELSE - 默认分支
// END - 条件表达式结束
MATCH (p:Person)
RETURN p.name,
       CASE
           WHEN p.age < 30 THEN '年轻人'
           WHEN p.age < 40 THEN '中年人'
           ELSE '资深人士'
       END AS `年龄段`;

// --- 16.6 coalesce - 返回第一个非空值 ---
// coalesce() - 函数，返回参数列表中第一个非空值
MATCH (p:Person)
RETURN p.name, coalesce(p.city, '未知城市') AS `城市`;

// ============================================================
// 第十七部分：完整示例 - 电影知识图谱查询
// ============================================================

// 17.1 创建完整的电影知识图谱 (重新构建演示数据)
// 注意：先清空数据库，避免数据重复
MATCH (n) DETACH DELETE n;

// 创建演员
// 使用变量名作为节点引用，后续可以直接使用这些变量名
CREATE (Alice:Person {name: 'Alice', age: 30}),
       (Bob:Person {name: 'Bob', age: 25}),
       (Charlie:Person {name: 'Charlie', age: 35}),
       (David:Person {name: 'David', age: 28}),
       (Eve:Person {name: 'Eve', age: 32});

// 创建电影
CREATE (Matrix:Movie {title: 'The Matrix', released: 1999, rating: 8.7}),
       (Inception:Movie {title: 'Inception', released: 2010, rating: 8.8}),
       (Interstellar:Movie {title: 'Interstellar', released: 2014, rating: 8.6}),
       (DarkKnight:Movie {title: 'The Dark Knight', released: 2008, rating: 9.0});

// 创建导演
CREATE (Lana:Director {name: 'Lana Wachowski', birthYear: 1965}),
       (Nolan:Director {name: 'Christopher Nolan', birthYear: 1970});

// 创建公司
CREATE (Warner:Company {name: 'Warner Bros', founded: 1923}),
       (Paramount:Company {name: 'Paramount', founded: 1912});

// 创建出演关系
MATCH (Alice:Person {name: 'Alice'}), (Matrix:Movie {title: 'The Matrix'})
CREATE (Alice)-[:ACTED_IN {roles: ['Neo']}]->(Matrix);

MATCH (Bob:Person {name: 'Bob'}), (Inception:Movie {title: 'Inception'})
CREATE (Bob)-[:ACTED_IN {roles: ['Cobb']}]->(Inception);

MATCH (Charlie:Person {name: 'Charlie'}), (Interstellar:Movie {title: 'Interstellar'})
CREATE (Charlie)-[:ACTED_IN {roles: ['Cooper']}]->(Interstellar);

MATCH (David:Person {name: 'David'}), (DarkKnight:Movie {title: 'The Dark Knight'})
CREATE (David)-[:ACTED_IN {roles: ['Joker']}]->(DarkKnight);

MATCH (Eve:Person {name: 'Eve'}), (Inception:Movie {title: 'Inception'})
CREATE (Eve)-[:ACTED_IN {roles: ['Ariadne']}]->(Inception);

// 创建导演关系
MATCH (Lana:Director {name: 'Lana Wachowski'}), (Matrix:Movie {title: 'The Matrix'})
CREATE (Lana)-[:DIRECTED {year: 1999}]->(Matrix);

MATCH (Nolan:Director {name: 'Christopher Nolan'})
CREATE (Nolan)-[:DIRECTED {year: 2010}]->(Inception),
       (Nolan)-[:DIRECTED {year: 2014}]->(Interstellar),
       (Nolan)-[:DIRECTED {year: 2008}]->(DarkKnight);

// 创建制作关系
MATCH (Matrix:Movie {title: 'The Matrix'}), (Warner:Company {name: 'Warner Bros'})
CREATE (Matrix)-[:PRODUCED_BY]->(Warner);

MATCH (Inception:Movie {title: 'Inception'}), (Warner:Company {name: 'Warner Bros'})
CREATE (Inception)-[:PRODUCED_BY]->(Warner);

MATCH (Interstellar:Movie {title: 'Interstellar'}), (Paramount:Company {name: 'Paramount'})
CREATE (Interstellar)-[:PRODUCED_BY]->(Paramount);

MATCH (DarkKnight:Movie {title: 'The Dark Knight'}), (Warner:Company {name: 'Warner Bros'})
CREATE (DarkKnight)-[:PRODUCED_BY]->(Warner);

// 17.2 综合查询：找出评分最高的导演及其作品
MATCH (d:Director)-[:DIRECTED]->(m:Movie)
RETURN d.name AS `导演`,
       collect(m.title) AS `作品`,
       avg(m.rating) AS `平均评分`,
       max(m.rating) AS `最高评分`
ORDER BY avg(m.rating) DESC;

// 17.3 查询每家公司制作的电影平均评分
MATCH (c:Company)<-[:PRODUCED_BY]-(m:Movie)
RETURN c.name AS `公司`,
       count(m) AS `电影数量`,
       avg(m.rating) AS `平均评分`
ORDER BY avg(m.rating) DESC;

// 17.4 查询演员合作关系图
MATCH (p1:Person)-[:ACTED_IN]->(m:Movie)<-[:ACTED_IN]-(p2:Person)
WHERE p1.name < p2.name
RETURN p1.name AS `演员 1`, p2.name AS `演员 2`, m.title AS `合作电影`;

// 17.5 统计每个导演的作品风格 (平均评分和数量)
// round() - 函数，四舍五入到指定小数位
MATCH (d:Director)-[:DIRECTED]->(m:Movie)
WITH d, count(m) AS movieCount, avg(m.rating) AS avgRating
RETURN d.name AS `导演`,
       movieCount AS `作品数`,
       round(avgRating * 10) / 10.0 AS `平均评分`
ORDER BY movieCount DESC, avgRating DESC;

// ============================================================
// 结束 - 以上语句可依次执行学习 Neo4j Cypher 语法
// ============================================================
